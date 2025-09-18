#!/usr/bin/env python3
"""
DPO Data Creation Script: Generate different model outputs using different random seeds
Directory structure: dpo_data/{dataset_name}/{case_name}/{seed}/
Based on create_dpo_data.py but using create_partcrafter_dpo_data.py address structure
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
import subprocess
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
from src.utils.eval_utils import (
    setup_gaps_tools
)
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

from huggingface_hub import snapshot_download
from accelerate.utils import set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NUM_PARTS = 16

class DPODataCreator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float,
                 build_gaps: bool = False,
                 load_trained_checkpoint: str = None,
                 checkpoint_iteration: int = None):
        """
        Initialize DPO data creator
        
        Args:
            model_path: Model weights path (base model for VAE, feature extractor, etc.)
            device: Computing device
            dtype: Data type
            build_gaps: Whether to build GAPS tools for evaluation
            load_trained_checkpoint: Path to trained checkpoint directory
            checkpoint_iteration: Checkpoint iteration number
        """
        self.device = device
        self.dtype = dtype
        self.build_gaps = build_gaps
        self.load_trained_checkpoint = load_trained_checkpoint
        self.checkpoint_iteration = checkpoint_iteration
        
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
        
        if load_trained_checkpoint is not None and checkpoint_iteration is not None:
            # Load trained checkpoint
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
            
            # Load transformer with EMA weights
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
    
    def load_gt_mesh(self, mesh_path: str) -> trimesh.Scene:
        """Load Ground Truth mesh"""
        mesh = trimesh.load(mesh_path, process=False)
        return mesh
    
    @torch.no_grad()
    def run_inference(self, 
                     image_path: str, 
                     num_parts: int,
                     seed: int = 0,
                     num_tokens: int = 1024,
                     num_inference_steps: int = 50,
                     guidance_scale: float = 7.0,
                     max_num_expanded_coords: int = 1e9,
                     use_flash_decoder: bool = False,
                     rmbg: bool = False,
                     rmbg_net: Any = None,
                     dtype: torch.dtype = torch.float16,
                     device: str = "cuda") -> Tuple[List[trimesh.Trimesh], Image.Image]:
        """
        Run PartCrafter inference with specific seed
        
        Returns:
            generated_meshes: List of generated meshes
            processed_image: Processed input image
        """
        assert 1 <= num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
        
        if rmbg:
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        else:
            img_pil = Image.open(image_path)
        
        start_time = time.time()
        
        # Set specific seed for this inference
        generator = torch.Generator(device=device).manual_seed(seed)
        
        outputs = self.pipeline(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        ).meshes
        
        end_time = time.time()
        logger.info(f"Time elapsed: {end_time - start_time:.2f} seconds")
        
        # Handle None outputs exactly like eval_scene.py
        for i in range(len(outputs)):
            if outputs[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        return outputs, img_pil
    
    def create_dpo_data_for_scene(self, 
                                scene_dir: Path,
                                output_dir: Path,
                                gt_dir: str,
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
        gt_mesh_path = Path(gt_dir) / f"{scene_name}.glb"
        if gt_mesh_path.exists():
            import shutil
            shutil.copy2(gt_mesh_path, scene_output_dir / "gt_mesh.glb")
        
        # Generate candidates with different seeds
        candidates = []
        
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
                pred_meshes, input_image = self.run_inference(
                    image_path=str(input_image_path),
                    num_parts=num_parts,
                    seed=seed,
                    num_tokens=inference_args.get('num_tokens', 1024) if inference_args else 1024,
                    num_inference_steps=inference_args.get('num_inference_steps', 50) if inference_args else 50,
                    guidance_scale=inference_args.get('guidance_scale', 7.0) if inference_args else 7.0,
                    max_num_expanded_coords=inference_args.get('max_num_expanded_coords', int(1e9)) if inference_args else int(1e9),
                    use_flash_decoder=inference_args.get('use_flash_decoder', False) if inference_args else False,
                    rmbg=inference_args.get('rmbg', False) if inference_args else False,
                    rmbg_net=self.rmbg_net,
                    dtype=self.dtype,
                    device=self.device
                )
                
                # Save individual meshes
                mesh_paths = []
                for i, mesh in enumerate(pred_meshes):
                    mesh_path = seed_dir / f"pred_obj_{i}.ply"
                    mesh.export(str(mesh_path))
                    mesh_paths.append(str(mesh_path))
                
                # Save merged mesh
                if pred_meshes:
                    merged_mesh = trimesh.util.concatenate(pred_meshes)
                    merged_path = seed_dir / "pred_merged.glb"
                    merged_mesh.export(str(merged_path))
                    
                    candidates.append({
                        'seed': seed,
                        'mesh_path': str(merged_path),
                        'individual_meshes': mesh_paths,
                        'num_parts': num_parts
                    })
                    
                    logger.info(f"Seed {seed}: Generated {len(pred_meshes)} meshes")
                
            except Exception as e:
                logger.error(f"Error generating with seed {seed}: {e}")
                continue
        
        # Save scene metadata
        scene_metadata = {
            'scene_name': scene_name,
            'num_parts': num_parts,
            'scene_info': scene_info,
            'seeds_used': seeds,
            'num_candidates': len(candidates)
        }
        
        metadata_file = scene_output_dir / "scene_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(scene_metadata, f, indent=2)
        
        logger.info(f"Scene {scene_name} completed: {len(candidates)} candidates")
        
        return scene_metadata
    
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
                    gt_dir=gt_dir,
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
            'scene_results': all_results
        }
        
        summary_file = output_path / "dpo_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"DPO dataset creation completed!")
        logger.info(f"Processed {len(all_results)}/{len(scene_dirs)} scenes")
        logger.info(f"Total candidates: {summary['total_candidates']}")
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
                       help='Path to trained checkpoint directory')
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
    parser.add_argument('--max_num_expanded_coords', type=int, default=1e9, help='Maximum number of expanded coordinates')
    parser.add_argument('--use_flash_decoder', action='store_true', help='Use flash decoder')
    parser.add_argument('--rmbg', action='store_true', help='Use background removal')
    parser.add_argument('--build_gaps', action='store_true', default=False,
                       help='Setup GAPS tools for evaluation')
    
    args = parser.parse_args()
    
    # Create DPO data creator
    creator = DPODataCreator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float,  # Use torch.float instead of torch.float16 for better memory management
        build_gaps=args.build_gaps,
        load_trained_checkpoint=args.load_trained_checkpoint,
        checkpoint_iteration=args.checkpoint_iteration
    )
    
    # Prepare inference arguments
    inference_args = {
        'num_tokens': args.num_tokens,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'max_num_expanded_coords': args.max_num_expanded_coords,
        'use_flash_decoder': args.use_flash_decoder,
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
