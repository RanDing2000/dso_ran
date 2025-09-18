#!/usr/bin/env python3
"""
DPO Data Creation Script for Real Scenes: Generate different model outputs using different random seeds
Directory structure: dpo_data/{dataset_name}/{scene_name}/{seed}/
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
import glob

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

MAX_NUM_PARTS = 16

class RealSceneDPODataCreator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 load_trained_checkpoint: str = None,
                 checkpoint_iteration: int = None):
        """
        Initialize DPO data creator for real scenes
        
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
        
        # Download and load base model components
        print(f"Downloading base model weights to: {model_path}")
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=model_path)
        
        # Download RMBG weights
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        
        # Init RMBG model for background removal
        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        self.rmbg_net.eval()
        
        print("Loading PartCrafter model...")
        
        if load_trained_checkpoint is not None and checkpoint_iteration is not None:
            # Load trained checkpoint
            print(f"Loading trained checkpoint from: {load_trained_checkpoint}")
            print(f"Checkpoint iteration: {checkpoint_iteration}")
            
            # Load base model components from pretrained weights
            vae = TripoSGVAEModel.from_pretrained(model_path, subfolder="vae")
            feature_extractor_dinov2 = BitImageProcessor.from_pretrained(model_path, subfolder="feature_extractor_dinov2")
            image_encoder_dinov2 = Dinov2Model.from_pretrained(model_path, subfolder="image_encoder_dinov2")
            noise_scheduler = RectifiedFlowScheduler.from_pretrained(model_path, subfolder="scheduler")
            
            # Load trained transformer from checkpoint
            checkpoint_path = os.path.join(load_trained_checkpoint, "checkpoints", f"{checkpoint_iteration:06d}")
            print(f"Loading transformer from: {checkpoint_path}")
            
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
            
            print(f"Loaded trained checkpoint successfully!")
            if loading_info:
                print(f"Loading info: {loading_info}")
        else:
            # Load standard pretrained model
            self.pipeline = PartCrafterPipeline.from_pretrained(model_path).to(device, dtype)
            print("Loaded standard pretrained model")
        
        print("Model loading completed!")
    
    def load_real_scene_config(self, scene_dir: str) -> Dict:
        """
        Load configuration for a real scene directory like demo-sc-1/
        
        Args:
            scene_dir: Path to scene directory (e.g., 'data/messy_kitchen_real/demo-sc-1/')
            
        Returns:
            Dictionary containing scene configuration
        """
        scene_dir = os.path.abspath(scene_dir)
        scene_name = os.path.basename(scene_dir.rstrip('/'))
        
        # Find images in the images/ subdirectory
        images_dir = os.path.join(scene_dir, 'images')
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        if not image_files:
            raise ValueError(f"No images found in: {images_dir}")
        
        # Use the first image as the main input image
        main_image = sorted(image_files)[0]
        
        # Count objects in scene-split directory
        scene_split_dir = os.path.join(scene_dir, 'scene-split')
        num_parts = 0
        if os.path.exists(scene_split_dir):
            ply_files = glob.glob(os.path.join(scene_split_dir, '*.ply'))
            num_parts = len(ply_files)
        
        # If no scene-split, try to estimate from scene.ply
        if num_parts == 0:
            scene_ply = os.path.join(scene_dir, 'scene.ply')
            if os.path.exists(scene_ply):
                # For now, assume a reasonable number of parts
                # In practice, you might want to analyze the scene.ply to count objects
                num_parts = 5  # Default assumption
        
        if num_parts == 0:
            raise ValueError(f"Could not determine number of parts in scene: {scene_dir}")
        
        config = {
            'scene_name': scene_name,
            'scene_dir': scene_dir,
            'image_path': main_image,
            'num_parts': num_parts,
            'scene_ply': os.path.join(scene_dir, 'scene.ply'),
            'scene_split_dir': scene_split_dir,
            'all_images': sorted(image_files),
            'valid': True
        }
        
        print(f"Loaded real scene config:")
        print(f"  Scene: {scene_name}")
        print(f"  Main image: {os.path.basename(main_image)}")
        print(f"  Number of parts: {num_parts}")
        print(f"  Total images: {len(image_files)}")
        
        return config
    
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
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        
        # Handle None outputs
        for i in range(len(outputs)):
            if outputs[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        return outputs, img_pil
    
    def create_single_scene_dpo_data(self, 
                                   scene_config: Dict,
                                   output_dir: str,
                                   seeds: List[int],
                                   inference_args: Dict = None) -> Dict[str, Any]:
        """
        Create DPO data for a single real scene with multiple seeds
        
        Returns:
            Dictionary containing generation results
        """
        scene_name = scene_config['scene_name']
        print(f"\nCreating DPO data for real scene: {scene_name}")
        print(f"Using seeds: {seeds}")
        
        try:
            # Create output directory structure: dpo_data/{dataset_name}/{scene_name}/
            scene_output_dir = os.path.join(output_dir, scene_name)
            os.makedirs(scene_output_dir, exist_ok=True)
            
            # Save input image to scene directory
            input_image_path = os.path.join(scene_output_dir, "input_image.jpg")
            input_image = Image.open(scene_config['image_path'])
            input_image.save(input_image_path)
            print(f"Input image saved to: {input_image_path}")
            
            # Process each seed
            all_results = []
            for seed in seeds:
                print(f"\n--- Processing seed: {seed} ---")
                
                # Set seed for this iteration
                set_seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Create seed-specific output directory
                seed_output_dir = os.path.join(scene_output_dir, str(seed))
                os.makedirs(seed_output_dir, exist_ok=True)
                
                # Run PartCrafter inference with specific seed
                pred_meshes, processed_image = self.run_inference(
                    image_path=scene_config['image_path'],
                    num_parts=scene_config['num_parts'],
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
                
                # Save the main merged mesh (no alignment needed for DPO)
                assert pred_meshes is not None
                pred_merged = trimesh.util.concatenate(pred_meshes)
                pred_merged.export(os.path.join(seed_output_dir, "pred_merged.glb"))
                print(f"Saved merged predicted mesh: pred_merged.glb")
                
                # Save individual predicted meshes
                if pred_meshes and len(pred_meshes) > 0:
                    # Save each predicted mesh as pred_obj_*.ply
                    print(f"Saving {len(pred_meshes)} predicted meshes as PLY files...")
                    for i, mesh in enumerate(pred_meshes):
                        ply_filename = f"pred_obj_{i}.ply"
                        ply_path = os.path.join(seed_output_dir, ply_filename)
                        mesh.export(ply_path)
                        print(f"  Saved: {ply_filename}")
                
                # Save seed info
                seed_info = {
                    'seed': seed,
                    'scene_name': scene_name,
                    'num_parts': scene_config['num_parts'],
                    'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'inference_args': inference_args or {},
                    'input_image_path': input_image_path,
                    'scene_dir': scene_config['scene_dir']
                }
                
                seed_info_path = os.path.join(seed_output_dir, "seed_info.json")
                with open(seed_info_path, 'w') as f:
                    json.dump(seed_info, f, indent=2)
                
                result = {
                    'scene_name': scene_name,
                    'num_parts': scene_config['num_parts'],
                    'seed': seed,
                    'mesh_dir': seed_output_dir,
                    'success': True,
                    'num_pred_meshes': len(pred_meshes)
                }
                
                all_results.append(result)
                print(f"Successfully created DPO data for scene: {scene_name} with seed: {seed}")
            
            # Save scene summary
            scene_summary = {
                'scene_name': scene_name,
                'scene_dir': scene_config['scene_dir'],
                'num_parts': scene_config['num_parts'],
                'seeds_processed': seeds,
                'total_generations': len(all_results),
                'successful_generations': len([r for r in all_results if r['success']]),
                'failed_generations': len([r for r in all_results if not r['success']]),
                'success_rate': len([r for r in all_results if r['success']]) / len(all_results) if all_results else 0.0,
                'detailed_results': all_results,
                'creation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            summary_path = os.path.join(scene_output_dir, 'scene_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(scene_summary, f, indent=2)
            
            print(f"\nScene {scene_name} completed. Summary saved to: {summary_path}")
            return scene_summary
            
        except Exception as e:
            print(f"Error creating DPO data for scene {scene_name}: {e}")
            return {
                'scene_name': scene_name,
                'num_parts': scene_config.get('num_parts', 0),
                'seeds_processed': seeds,
                'error': str(e),
                'success': False
            }
    
    def create_dpo_dataset_from_real_scenes(self, 
                                          scene_dirs: List[str],
                                          output_base_dir: str,
                                          seeds: List[int],
                                          inference_args: Dict = None) -> Dict[str, Any]:
        """
        Create DPO dataset from multiple real scene directories
        
        Args:
            scene_dirs: List of scene directory paths
            output_base_dir: Base directory for DPO data
            seeds: List of random seeds to use
            inference_args: Inference parameters
            
        Returns:
            Dictionary containing all generation results
        """
        print(f"Starting DPO dataset creation from {len(scene_dirs)} real scenes")
        print(f"Using seeds: {seeds}")
        
        # Create output directory structure: dpo_data/{dataset_name}/
        dataset_name = "real_scenes"
        output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        # Process each scene
        all_results = []
        for scene_idx, scene_dir in enumerate(scene_dirs):
            print(f"\n{'='*80}")
            print(f"Processing scene {scene_idx + 1}/{len(scene_dirs)}: {scene_dir}")
            print(f"{'='*80}")
            
            try:
                # Load scene configuration
                scene_config = self.load_real_scene_config(scene_dir)
                
                # Create DPO data for this scene
                scene_result = self.create_single_scene_dpo_data(
                    scene_config, output_dir, seeds, inference_args
                )
                all_results.append(scene_result)
                
            except Exception as e:
                print(f"Error processing scene {scene_dir}: {e}")
                all_results.append({
                    'scene_dir': scene_dir,
                    'error': str(e),
                    'success': False
                })
        
        # Save overall summary
        summary_path = os.path.join(output_dir, 'dpo_summary.json')
        successful_results = [r for r in all_results if r.get('success', False)]
        failed_results = [r for r in all_results if not r.get('success', False)]
        
        summary = {
            'dataset_name': dataset_name,
            'seeds_used': seeds,
            'total_scenes': len(scene_dirs),
            'total_generations': sum([r.get('total_generations', 0) for r in all_results]),
            'successful_generations': sum([r.get('successful_generations', 0) for r in all_results]),
            'failed_generations': sum([r.get('failed_generations', 0) for r in all_results]),
            'success_rate': sum([r.get('successful_generations', 0) for r in all_results]) / max(sum([r.get('total_generations', 0) for r in all_results]), 1),
            'per_scene_summary': {r.get('scene_name', f'scene_{i}'): r for i, r in enumerate(all_results)},
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDPO dataset creation completed!")
        print(f"Dataset: {dataset_name}")
        print(f"Total scenes processed: {len(scene_dirs)}")
        print(f"Seeds processed: {seeds}")
        print(f"Total generations: {summary['total_generations']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Results saved to: {output_dir}")
        
        return {
            'summary': summary,
            'results': all_results,
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(description='PartCrafter DPO data creation script for real scenes')
    parser.add_argument('--scene_dirs', type=str, nargs='+', 
                       default=['data/messy_kitchen_real/demo-sc-1'],
                       help='List of scene directory paths')
    parser.add_argument('--output_base_dir', type=str, 
                       default='./dpo_data',
                       help='Base directory for DPO data output')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter',
                       help='Base model weights path')
    parser.add_argument('--load_trained_checkpoint', type=str, 
                       default='runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512',
                       help='Path to trained checkpoint directory')
    parser.add_argument('--checkpoint_iteration', type=int, default=17000,
                       help='Checkpoint iteration number')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                       help='Random seeds to use for generation')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Number of tokens for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0, help='Guidance scale for generation')
    parser.add_argument('--max_num_expanded_coords', type=int, default=1e9, help='Maximum number of expanded coordinates')
    parser.add_argument('--use_flash_decoder', action='store_true', help='Use flash decoder')
    parser.add_argument('--rmbg', action='store_true', help='Use background removal')
    
    args = parser.parse_args()
    
    # Create DPO data creator
    creator = RealSceneDPODataCreator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float16,
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
    results = creator.create_dpo_dataset_from_real_scenes(
        scene_dirs=args.scene_dirs,
        output_base_dir=args.output_base_dir,
        seeds=args.seeds,
        inference_args=inference_args
    )
    
    print(f"\nDPO data creation completed! Results saved to: {args.output_base_dir}")

if __name__ == "__main__":
    main()
