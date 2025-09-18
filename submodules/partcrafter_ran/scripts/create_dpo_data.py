#!/usr/bin/env python3
"""
DPO Data Creation Script: Generate different model outputs using different random seeds
Directory structure: dpo_data/{dataset_name}/{case_name}/{seed}/
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

MAX_NUM_PARTS = 16

class DPODataCreator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
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
    
    def load_test_config(self, config_path: str) -> List[Dict]:
        """Load test configuration"""
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs
    
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
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        
        # Handle None outputs
        for i in range(len(outputs)):
            if outputs[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        return outputs, img_pil
    
    def create_single_case_dpo_data(self, 
                                  config: Dict,
                                  output_dir: str,
                                  seed: int,
                                  inference_args: Dict = None) -> Dict[str, Any]:
        """
        Create DPO data for a single test case with specific seed
        
        Returns:
            Dictionary containing generation results
        """
        case_name = Path(config['mesh_path']).stem
        print(f"\nCreating DPO data for case: {case_name} with seed: {seed}")
        
        try:
            # Load GT mesh
            gt_mesh = self.load_gt_mesh(config['mesh_path'])
            
            # Create output directory structure: dpo_data/{dataset_name}/{case_name}/{seed}/
            case_output_dir = os.path.join(output_dir, case_name, str(seed))
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Run PartCrafter inference with specific seed
            pred_meshes, input_image = self.run_inference(
                image_path=config['image_path'],
                num_parts=config['num_parts'],
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
            
            # Extract GT meshes for alignment only
            if isinstance(gt_mesh, trimesh.Scene):
                gt_meshes = list(gt_mesh.geometry.values())
            elif isinstance(gt_mesh, trimesh.Trimesh):
                gt_meshes = [gt_mesh]
            else:
                gt_meshes = []
            
            # Save the main merged mesh (no alignment needed for DPO)
            assert pred_meshes is not None
            pred_merged = trimesh.util.concatenate(pred_meshes)
            pred_merged.export(os.path.join(case_output_dir, "pred_merged.glb"))
            print(f"Saved merged predicted mesh: pred_merged.glb")
            
            # Save input image to scene directory (parent directory)
            scene_dir = os.path.dirname(case_output_dir)  # Go up one level to scene directory
            input_image_path = os.path.join(scene_dir, "input_image.jpg")
            input_image.save(input_image_path)
            print(f"Input image saved to: {input_image_path}")
            
            # Save individual predicted meshes
            if pred_meshes and len(pred_meshes) > 0:
                # Save each predicted mesh as pred_obj_*.ply
                print(f"Saving {len(pred_meshes)} predicted meshes as PLY files...")
                for i, mesh in enumerate(pred_meshes):
                    ply_filename = f"pred_obj_{i}.ply"
                    ply_path = os.path.join(case_output_dir, ply_filename)
                    mesh.export(ply_path)
                    print(f"  Saved: {ply_filename}")
            
            # Save seed info
            seed_info = {
                'seed': seed,
                'case_name': case_name,
                'num_parts': config['num_parts'],
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'inference_args': inference_args or {},
                'input_image_path': input_image_path
            }
            
            seed_info_path = os.path.join(case_output_dir, "seed_info.json")
            with open(seed_info_path, 'w') as f:
                json.dump(seed_info, f, indent=2)
            
            result = {
                'case_name': case_name,
                'num_parts': config['num_parts'],
                'seed': seed,
                'mesh_dir': case_output_dir,
                'success': True,
                'num_pred_meshes': len(pred_meshes),
                'num_gt_meshes': len(gt_meshes)
            }
            
            print(f"Successfully created DPO data for case: {case_name} with seed: {seed}")
            return result
            
        except Exception as e:
            print(f"Error creating DPO data for case {case_name} with seed {seed}: {e}")
            return {
                'case_name': case_name,
                'num_parts': config.get('num_parts', 0),
                'seed': seed,
                'error': str(e),
                'success': False
            }
    
    def create_dpo_dataset(self, 
                          config_path: str,
                          output_base_dir: str,
                          seeds: List[int],
                          inference_args: Dict = None) -> Dict[str, Any]:
        """
        Create DPO dataset with multiple seeds
        
        Args:
            config_path: Path to test configuration file
            output_base_dir: Base directory for DPO data
            seeds: List of random seeds to use
            inference_args: Inference parameters
            
        Returns:
            Dictionary containing all generation results
        """
        print(f"Starting DPO dataset creation: {config_path}")
        print(f"Using seeds: {seeds}")
        
        # Load test configuration
        configs = self.load_test_config(config_path)
        print(f"Found {len(configs)} test cases")
        
        # Extract dataset name from config path
        dataset_name = Path(config_path).stem
        
        # Sort configs by num_parts (descending) to prioritize scenes with more objects
        configs_sorted = sorted(configs, key=lambda x: x.get('num_parts', 0), reverse=True)
        print(f"Processing all {len(configs_sorted)} scenes, sorted by object count (descending)")
        
        # Print top 10 scenes with most objects
        print("Top 10 scenes with most objects:")
        for i, config in enumerate(configs_sorted[:10]):
            case_name = Path(config['mesh_path']).stem
            num_parts = config.get('num_parts', 0)
            print(f"  {i+1}. {case_name}: {num_parts} objects")
        
        # Create output directory structure: dpo_data/{dataset_name}/
        output_dir = os.path.join(output_base_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output directory: {output_dir}")
        
        # Create DPO data for each scene and seed
        all_results = []
        total_scenes = len(configs_sorted)
        
        for scene_idx, config in enumerate(configs_sorted):
            case_name = Path(config['mesh_path']).stem
            num_parts = config.get('num_parts', 0)
            print(f"\n{'='*80}")
            print(f"Processing scene {scene_idx + 1}/{total_scenes}: {case_name} ({num_parts} objects)")
            print(f"{'='*80}")
            
            # Process this scene with all seeds
            scene_results = []
            for seed in seeds:
                print(f"\n--- Processing seed: {seed} ---")
                
                # Set seed for this iteration
                set_seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Create DPO data for this scene and seed
                result = self.create_single_case_dpo_data(
                    config, output_dir, seed, inference_args
                )
                scene_results.append(result)
                all_results.append(result)
            
            # Save scene-specific results
            scene_results_path = os.path.join(output_dir, f'scene_{scene_idx+1:03d}_{case_name}_results.json')
            with open(scene_results_path, 'w') as f:
                json.dump({
                    'scene_index': scene_idx + 1,
                    'case_name': case_name,
                    'num_parts': num_parts,
                    'total_seeds': len(seeds),
                    'successful_seeds': len([r for r in scene_results if r['success']]),
                    'failed_seeds': len([r for r in scene_results if not r['success']]),
                    'detailed_results': scene_results
                }, f, indent=2)
            
            print(f"Scene {scene_idx + 1} completed. Results saved to: {scene_results_path}")
        
        # Save overall summary
        summary_path = os.path.join(output_dir, 'dpo_summary.json')
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]
        
        summary = {
            'dataset_name': dataset_name,
            'seeds_used': seeds,
            'total_scenes': len(configs_sorted),
            'total_generations': len(all_results),
            'successful_generations': len(successful_results),
            'failed_generations': len(failed_results),
            'success_rate': len(successful_results) / len(all_results) if all_results else 0.0,
            'per_seed_summary': {},
            'per_scene_summary': {}
        }
        
        # Add per-seed summary
        for seed in seeds:
            seed_results = [r for r in all_results if r['seed'] == seed]
            seed_successful = [r for r in seed_results if r['success']]
            summary['per_seed_summary'][str(seed)] = {
                'total_scenes': len(seed_results),
                'successful_scenes': len(seed_successful),
                'failed_scenes': len(seed_results) - len(seed_successful),
                'success_rate': len(seed_successful) / len(seed_results) if seed_results else 0.0
            }
        
        # Add per-scene summary
        for scene_idx, config in enumerate(configs_sorted):
            case_name = Path(config['mesh_path']).stem
            num_parts = config.get('num_parts', 0)
            scene_results = [r for r in all_results if r['case_name'] == case_name]
            scene_successful = [r for r in scene_results if r['success']]
            summary['per_scene_summary'][case_name] = {
                'scene_index': scene_idx + 1,
                'num_parts': num_parts,
                'total_seeds': len(scene_results),
                'successful_seeds': len(scene_successful),
                'failed_seeds': len(scene_results) - len(scene_successful),
                'success_rate': len(scene_successful) / len(scene_results) if scene_results else 0.0
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDPO dataset creation completed!")
        print(f"Dataset: {dataset_name}")
        print(f"Total scenes processed: {len(configs_sorted)}")
        print(f"Seeds processed: {seeds}")
        print(f"Total generations: {len(all_results)}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Results saved to: {output_dir}")
        
        # Print summary statistics
        print(f"\nSummary Statistics:")
        print(f"  Scenes with most objects:")
        for i, config in enumerate(configs_sorted[:5]):
            case_name = Path(config['mesh_path']).stem
            num_parts = config.get('num_parts', 0)
            print(f"    {i+1}. {case_name}: {num_parts} objects")
        
        print(f"  Per-seed success rates:")
        for seed in seeds:
            seed_stats = summary['per_seed_summary'][str(seed)]
            print(f"    Seed {seed}: {seed_stats['success_rate']:.2%} ({seed_stats['successful_scenes']}/{seed_stats['total_scenes']})")
        
        return {
            'summary': summary,
            'results': all_results,
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(description='PartCrafter DPO data creation script')
    parser.add_argument('--config_path', type=str, 
                       default='data/preprocessed_data_messy_kitchen_scenes_part2/messy_kitchen_test_100.json',
                       help='Test configuration file path')
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
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 42, 123, 456, 789],
                       help='Random seeds to use for generation')
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
        dtype=torch.float16,
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
        config_path=args.config_path,
        output_base_dir=args.output_base_dir,
        seeds=args.seeds,
        inference_args=inference_args
    )
    
    print(f"\nDPO data creation completed! Results saved to: {args.output_base_dir}")

if __name__ == "__main__":
    main()
