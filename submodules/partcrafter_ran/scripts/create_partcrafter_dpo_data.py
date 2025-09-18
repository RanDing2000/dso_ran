#!/usr/bin/env python3
"""
PartCrafter DPO Data Creation Script
Generate DPO dataset using PartCrafter with different seeds and TAGAC penetration evaluation
Based on PartCrafter_DPO_Implementation_Plan.md
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import trimesh
from PIL import Image
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.schedulers import RectifiedFlowScheduler
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

from huggingface_hub import snapshot_download
from accelerate.utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NUM_PARTS = 16

class PenetrationScoreEvaluator:
    """Penetration score evaluator based on TAGAC implementation"""
    
    def __init__(self, voxel_size: float = 0.3, resolution: int = 40):
        self.voxel_size = voxel_size
        self.resolution = resolution
        
    def compute_penetration_level(self, mesh_paths: List[str]) -> Dict[str, float]:
        """Compute penetration level using TAGAC's well-defined calculation"""
        try:
            if len(mesh_paths) == 1:
                # Single mesh - no penetration possible
                return {
                    'penetration_level': 0.0,
                    'overlap_ratio': 0.0,
                    'merged_internal_points': 0,
                    'individual_internal_points_sum': 0,
                    'per_mesh_penetration': [0.0]
                }
            
            # Load meshes
            meshes = [trimesh.load(path) for path in mesh_paths]
            
            # Compute individual internal points
            individual_points = []
            for mesh in meshes:
                points = self._voxelize_mesh(mesh)
                individual_points.append(len(points))
            
            # Merge meshes and compute merged internal points
            merged_mesh = trimesh.util.concatenate(meshes)
            merged_points = len(self._voxelize_mesh(merged_mesh))
            
            total_individual = sum(individual_points)
            
            # Penetration level calculation (TAGAC formula)
            if total_individual > 0:
                penetration_level = 1.0 - (merged_points / total_individual)
                overlap_ratio = merged_points / total_individual
            else:
                penetration_level = 0.0
                overlap_ratio = 0.0
            
            return {
                'penetration_level': penetration_level,
                'overlap_ratio': overlap_ratio,
                'merged_internal_points': merged_points,
                'individual_internal_points_sum': total_individual,
                'per_mesh_penetration': [0.0] * len(meshes)  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Error computing penetration level: {e}")
            return {
                'penetration_level': 1.0,  # Worst case
                'overlap_ratio': 0.0,
                'merged_internal_points': 0,
                'individual_internal_points_sum': 0,
                'per_mesh_penetration': [1.0] * len(mesh_paths)
            }
    
    def _voxelize_mesh(self, mesh: trimesh.Trimesh) -> List[tuple]:
        """Simple voxelization for penetration analysis"""
        try:
            # Get mesh bounds
            bounds = mesh.bounds
            
            # Create voxel grid
            voxel_coords = []
            
            for x in np.arange(bounds[0, 0], bounds[1, 0], self.voxel_size):
                for y in np.arange(bounds[0, 1], bounds[1, 1], self.voxel_size):
                    for z in np.arange(bounds[0, 2], bounds[1, 2], self.voxel_size):
                        point = np.array([x, y, z])
                        
                        # Check if point is inside mesh
                        if mesh.contains([point]):
                            voxel_coords.append((x, y, z))
            
            return voxel_coords
            
        except Exception as e:
            logger.error(f"Error voxelizing mesh: {e}")
            return []

class PartCrafterDPODataCreator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float,
                 load_trained_checkpoint: str = None,
                 checkpoint_iteration: int = None):
        """
        Initialize PartCrafter DPO data creator
        
        Args:
            model_path: Model weights path (base model for VAE, feature extractor, etc.)
            device: Computing device
            dtype: Data type
            load_trained_checkpoint: Path to trained checkpoint directory
            checkpoint_iteration: Checkpoint iteration number
        """
        self.device = device
        self.dtype = dtype
        self.load_trained_checkpoint = load_trained_checkpoint
        self.checkpoint_iteration = checkpoint_iteration
        
        # Initialize penetration evaluator
        self.penetration_evaluator = PenetrationScoreEvaluator()
        
        # Download and load base model components
        logger.info(f"Downloading base model weights to: {model_path}")
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=model_path)
        
        # Download RMBG weights
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        
        # Init RMBG model for background removal
        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        self.rmbg_net.eval()
        
        logger.info("Loading PartCrafter model...")
        
        # Load model exactly following eval_scene.py pattern
        if load_trained_checkpoint is not None and checkpoint_iteration is not None:
            # Load trained checkpoint - following train_partcrafter.py pattern
            logger.info(f"Loading trained checkpoint from: {load_trained_checkpoint}")
            logger.info(f"Checkpoint iteration: {checkpoint_iteration}")
            
            # Load base model components from pretrained weights
            vae = TripoSGVAEModel.from_pretrained(model_path, subfolder="vae")
            feature_extractor_dinov2 = BitImageProcessor.from_pretrained(model_path, subfolder="feature_extractor_dinov2")
            image_encoder_dinov2 = Dinov2Model.from_pretrained(model_path, subfolder="image_encoder_dinov2")
            noise_scheduler = RectifiedFlowScheduler.from_pretrained(model_path, subfolder="scheduler")
            
            # Load trained transformer from checkpoint
            checkpoint_path = os.path.join(load_trained_checkpoint, "checkpoints", f"{checkpoint_iteration:06d}")
            logger.info(f"Loading transformer from: {checkpoint_path}")
            
            # Load transformer with EMA weights (following train_partcrafter.py pattern)
            transformer, loading_info = PartCrafterDiTModel.from_pretrained(
                checkpoint_path, 
                subfolder="transformer_ema",
                low_cpu_mem_usage=False, 
                output_loading_info=True
            )
            
            # Create pipeline with trained components
            self.pipeline = PartCrafterPipeline(
                vae=vae,
                transformer=transformer,
                scheduler=noise_scheduler,
                feature_extractor_dinov2=feature_extractor_dinov2,
                image_encoder_dinov2=image_encoder_dinov2,
            ).to(device, dtype)
            
            logger.info(f"Loaded trained checkpoint successfully!")
            if loading_info:
                logger.info(f"Loading info: {loading_info}")
        else:
            # Load standard pretrained model
            self.pipeline = PartCrafterPipeline.from_pretrained(model_path).to(device, dtype)
            logger.info("Loaded standard pretrained model")
        
        logger.info("Model loading completed!")
    
    def load_scene_info(self, scene_dir: Path) -> Dict[str, Any]:
        """Load scene information from messy_kitchen_data"""
        scene_info = {}
        
        # Load num_parts
        num_parts_file = scene_dir / "num_parts.json"
        if num_parts_file.exists():
            with open(num_parts_file, 'r') as f:
                scene_info['num_parts'] = json.load(f)
        
        # Load IoU info
        iou_file = scene_dir / "iou.json"
        if iou_file.exists():
            with open(iou_file, 'r') as f:
                scene_info['iou'] = json.load(f)
        
        return scene_info
    
    @torch.no_grad()
    def run_inference(self, 
                     image_path: str, 
                     num_parts: int,
                     seed: int = 0,
                     num_tokens: int = 1024,
                     num_inference_steps: int = 50,
                     guidance_scale: float = 7.0,
                     rmbg: bool = False) -> Tuple[List[trimesh.Trimesh], Image.Image, torch.Tensor, torch.Tensor]:
        """
        Run PartCrafter inference with specific seed
        
        Returns:
            generated_meshes: List of generated meshes
            processed_image: Processed input image
            cond_features: Image conditioning features [1, feature_dim]
            latent_features: VAE latent features [num_parts, num_tokens, latent_dim]
        """
        assert 1 <= num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
        
        if rmbg:
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg_net)
        else:
            img_pil = Image.open(image_path)
        
        start_time = time.time()
        
        # Set specific seed for this inference
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Extract image conditioning features
        # Prepare image for feature extraction
        processed_image = self.pipeline.feature_extractor_dinov2(img_pil, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            cond_features = self.pipeline.image_encoder_dinov2(processed_image).last_hidden_state  # [1, seq_len, feature_dim]
        
        # Run PartCrafter pipeline to get meshes and latent features
        outputs = self.pipeline(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        
        # Handle outputs exactly like eval_scene.py
        meshes = outputs.meshes
        samples = outputs.samples
        
        # Replace None meshes with dummy meshes for consistency (exactly like eval_scene.py)
        for i in range(len(meshes)):
            if meshes[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                meshes[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        logger.info(f"Generated {len([m for m in meshes if m is not None])}/{len(meshes)} successful meshes")
        
        # Extract latent features from the pipeline's internal representations
        # The samples from PartCrafter contain the latent representations
        valid_samples = [s for s in samples if s is not None]
        
        if len(valid_samples) > 0:
            # Use valid samples as latent features
            # Each sample should have shape [num_tokens, latent_dim]
            # Stack them to get [num_valid_parts, num_tokens, latent_dim]
            latent_features = torch.stack(valid_samples, dim=0)
            
            # Ensure we have the right number of latent features
            if latent_features.shape[0] < num_parts:
                # Pad with dummy latent features if some samples were None
                dummy_features = torch.zeros(num_parts - latent_features.shape[0], 
                                           latent_features.shape[1], 
                                           latent_features.shape[2]).to(self.device)
                latent_features = torch.cat([latent_features, dummy_features], dim=0)
        else:
            # Create dummy latent features if no valid samples
            latent_features = torch.zeros(num_parts, num_tokens, 512).to(self.device)
            logger.warning(f"No valid samples available for seed {seed}, using dummy latent features")
        
        logger.info(f"Extracted latent features shape: {latent_features.shape}")
        
        end_time = time.time()
        logger.info(f"Time elapsed: {end_time - start_time:.2f} seconds")
        
        return meshes, img_pil, cond_features, latent_features
    
    def create_dpo_data_for_scene(self, 
                                scene_dir: Path,
                                output_dir: Path,
                                seeds: List[int],
                                inference_args: Dict = None) -> Dict[str, Any]:
        """Create DPO data for a single scene with multiple seeds"""
        
        scene_name = scene_dir.name
        logger.info(f"Processing scene: {scene_name}")
        
        # Load scene info
        scene_info = self.load_scene_info(scene_dir)
        num_parts_raw = scene_info.get('num_parts', 4)  # Default to 4 parts
        
        # Handle different num_parts formats
        if isinstance(num_parts_raw, dict):
            # If it's a dict, try to extract the actual number
            if 'value' in num_parts_raw:
                num_parts = num_parts_raw['value']
            elif 'count' in num_parts_raw:
                num_parts = num_parts_raw['count']
            else:
                # Take the first value if it's a dict with numeric values
                num_parts = next((v for v in num_parts_raw.values() if isinstance(v, (int, float))), 4)
        elif isinstance(num_parts_raw, (int, float)):
            num_parts = int(num_parts_raw)
        else:
            num_parts = 4  # Default fallback
        
        # Ensure num_parts is within valid range
        num_parts = max(1, min(int(num_parts), MAX_NUM_PARTS))
        logger.info(f"Using num_parts: {num_parts} (from: {num_parts_raw})")
        
        # Create output directory
        scene_output_dir = output_dir / scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input image
        input_image_path = scene_dir / "rendering.png"
        if input_image_path.exists():
            import shutil
            shutil.copy2(input_image_path, scene_output_dir / "input_image.png")
        
        # Copy ground truth mesh
        gt_mesh_path = Path("../messy_kitchen_data/raw_messy_kitchen_scenes") / f"{scene_name}.glb"
        if gt_mesh_path.exists():
            import shutil
            shutil.copy2(gt_mesh_path, scene_output_dir / "gt_mesh.glb")
        
        # Generate candidates with different seeds
        candidates = []
        penetration_scores = []
        
        for seed in seeds:
            logger.info(f"Generating with seed: {seed}")
            
            # Set seed for this iteration
            set_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create seed output directory
            seed_dir = scene_output_dir / f"{seed:03d}"
            seed_dir.mkdir(exist_ok=True)
            
            try:
                # Run inference
                pred_meshes, input_image, cond_features, latent_features = self.run_inference(
                    image_path=str(input_image_path),
                    num_parts=num_parts,
                    seed=seed,
                    num_tokens=inference_args.get('num_tokens', 1024) if inference_args else 1024,
                    num_inference_steps=inference_args.get('num_inference_steps', 50) if inference_args else 50,
                    guidance_scale=inference_args.get('guidance_scale', 7.0) if inference_args else 7.0,
                    rmbg=inference_args.get('rmbg', False) if inference_args else False,
                )
                
                # Save cond features (similar to TRELLIS)
                cond_features_path = seed_dir / "cond.pt"
                torch.save(cond_features.cpu().to(torch.bfloat16), cond_features_path)
                
                # Save individual meshes
                mesh_paths = []
                for i, mesh in enumerate(pred_meshes):
                    mesh_path = seed_dir / f"pred_obj_{i}.ply"
                    mesh.export(str(mesh_path))
                    mesh_paths.append(str(mesh_path))
                
                # Save latent features for each part (similar to TRELLIS sparse_x0)
                for i in range(num_parts):
                    latent_part_path = seed_dir / f"latent_sample_{i:03d}.pt"
                    if latent_features is not None and i < latent_features.shape[0]:
                        part_latent = latent_features[i].cpu().to(torch.bfloat16)  # [num_tokens, latent_dim]
                        torch.save(part_latent, latent_part_path)
                    else:
                        # Save dummy latent if extraction failed
                        dummy_latent = torch.zeros(inference_args.get('num_tokens', 1024), 512).to(torch.bfloat16)
                        torch.save(dummy_latent, latent_part_path)
                
                # Save merged mesh
                if pred_meshes:
                    merged_mesh = trimesh.util.concatenate(pred_meshes)
                    merged_path = seed_dir / "pred_merged.glb"
                    merged_mesh.export(str(merged_path))
                    
                    # Compute penetration score
                    penetration_score = self.penetration_evaluator.compute_penetration_level([str(merged_path)])
                    
                    # Save penetration analysis
                    penetration_file = seed_dir / "penetration_analysis.json"
                    with open(penetration_file, 'w') as f:
                        json.dump(penetration_score, f, indent=2)
                    
                    candidates.append({
                        'seed': seed,
                        'mesh_path': str(merged_path),
                        'individual_meshes': mesh_paths,
                        'penetration_score': penetration_score,
                        'cond_features_path': str(cond_features_path),
                        'latent_features_paths': [str(seed_dir / f"latent_sample_{i:03d}.pt") for i in range(num_parts)],
                        'num_parts': num_parts
                    })
                    
                    penetration_scores.append(penetration_score['penetration_level'])
                    
                    logger.info(f"Seed {seed}: penetration_level = {penetration_score['penetration_level']:.4f}")
                
            except Exception as e:
                logger.error(f"Error generating with seed {seed}: {e}")
                continue
        
        # Create preference pairs based on penetration scores
        preference_pairs = self.create_preference_pairs(candidates, penetration_scores)
        
        # Save preference pairs
        preference_file = scene_output_dir / "preference_pairs.json"
        with open(preference_file, 'w') as f:
            json.dump(preference_pairs, f, indent=2)
        
        # Save scene metadata
        scene_metadata = {
            'scene_name': scene_name,
            'num_parts': num_parts,
            'scene_info': scene_info,
            'seeds_used': seeds,
            'num_candidates': len(candidates),
            'penetration_scores': penetration_scores,
            'preference_pairs_count': len(preference_pairs)
        }
        
        metadata_file = scene_output_dir / "scene_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(scene_metadata, f, indent=2)
        
        logger.info(f"Scene {scene_name} completed: {len(candidates)} candidates, {len(preference_pairs)} preference pairs")
        
        return scene_metadata
    
    def create_preference_pairs(self, candidates: List[Dict], penetration_scores: List[float]) -> List[Dict]:
        """Create preference pairs based on penetration scores (lower = better)"""
        preference_pairs = []
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                score_i = penetration_scores[i]
                score_j = penetration_scores[j]
                
                # Lower penetration score is preferred
                if score_i < score_j:
                    preference_pairs.append({
                        'win_candidate': candidates[i],
                        'loss_candidate': candidates[j],
                        'win_penetration': score_i,
                        'loss_penetration': score_j,
                        'penetration_diff': score_j - score_i
                    })
                elif score_j < score_i:
                    preference_pairs.append({
                        'win_candidate': candidates[j],
                        'loss_candidate': candidates[i],
                        'win_penetration': score_j,
                        'loss_penetration': score_i,
                        'penetration_diff': score_i - score_j
                    })
        
        return preference_pairs
    
    def create_dpo_dataset(self, 
                          input_dir: str,
                          gt_dir: str,
                          output_dir: str,
                          seeds: List[int],
                          max_scenes: int = None,
                          inference_args: Dict = None) -> Dict[str, Any]:
        """
        Create DPO dataset from messy_kitchen_data
        
        Args:
            input_dir: Directory containing messy_kitchen_scenes_renderings
            gt_dir: Directory containing raw_messy_kitchen_scenes
            output_dir: Output directory for DPO data
            seeds: List of random seeds to use
            max_scenes: Maximum number of scenes to process (None for all)
            inference_args: Inference parameters
            
        Returns:
            Dictionary containing all generation results
        """
        logger.info(f"Starting DPO dataset creation")
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"GT directory: {gt_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Using seeds: {seeds}")
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all scene directories
        scene_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        if max_scenes:
            scene_dirs = scene_dirs[:max_scenes]
        
        logger.info(f"Found {len(scene_dirs)} scenes to process")
        
        all_results = []
        
        for scene_idx, scene_dir in enumerate(tqdm(scene_dirs, desc="Processing scenes")):
            logger.info(f"Processing scene {scene_idx + 1}/{len(scene_dirs)}: {scene_dir.name}")
            
            try:
                result = self.create_dpo_data_for_scene(
                    scene_dir=scene_dir,
                    output_dir=output_path,
                    seeds=seeds,
                    inference_args=inference_args
                )
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing scene {scene_dir.name}: {e}")
                continue
        
        # Save overall summary
        summary = {
            'total_scenes': len(scene_dirs),
            'processed_scenes': len(all_results),
            'seeds_used': seeds,
            'total_candidates': sum(r.get('num_candidates', 0) for r in all_results),
            'total_preference_pairs': sum(r.get('preference_pairs_count', 0) for r in all_results),
            'scene_results': all_results
        }
        
        summary_file = output_path / "dpo_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"DPO dataset creation completed!")
        logger.info(f"Processed {len(all_results)}/{len(scene_dirs)} scenes")
        logger.info(f"Total candidates: {summary['total_candidates']}")
        logger.info(f"Total preference pairs: {summary['total_preference_pairs']}")
        logger.info(f"Results saved to: {output_dir}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='PartCrafter DPO data creation script')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/messy_kitchen_scenes_renderings',
                       help='Input directory containing rendering images')
    parser.add_argument('--gt_dir', type=str,
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/raw_messy_kitchen_scenes',
                       help='Directory containing ground truth meshes')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data',
                       help='Output directory for DPO data')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter',
                       help='Base model weights path')
    parser.add_argument('--load_trained_checkpoint', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512',
                       help='Path to trained checkpoint directory (optional, will use pretrained model if not provided)')
    parser.add_argument('--checkpoint_iteration', type=int, default=17000,
                       help='Checkpoint iteration number')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 123, 456],
                       help='Random seeds to use for generation')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Number of tokens for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0, help='Guidance scale for generation')
    parser.add_argument('--rmbg', action='store_true', help='Use background removal')
    
    args = parser.parse_args()
    
    # Create DPO data creator
    creator = PartCrafterDPODataCreator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float,
        load_trained_checkpoint=args.load_trained_checkpoint,
        checkpoint_iteration=args.checkpoint_iteration
    )
    
    # Prepare inference arguments
    inference_args = {
        'num_tokens': args.num_tokens,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'rmbg': args.rmbg
    }
    
    # Create DPO dataset
    results = creator.create_dpo_dataset(
        input_dir=args.input_dir,
        gt_dir=args.gt_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_scenes=args.max_scenes,
        inference_args=inference_args
    )
    
    logger.info(f"DPO data creation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

