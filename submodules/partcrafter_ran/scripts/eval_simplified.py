#!/usr/bin/env python3
"""
Simplified PartCrafter evaluation script using utility functions
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

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.eval_utils import (
    setup_gaps_tools, compute_aligned_metrics, 
    save_meshes_with_alignment
)
from src.utils.visualization_utils import (
    render_comparison_with_alignment, create_evaluation_visualizations,
    print_evaluation_summary, print_case_results
)
from huggingface_hub import snapshot_download


class PartCrafterEvaluator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter-Scene",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 build_gaps: bool = True):
        """Initialize PartCrafter evaluator"""
        self.device = device
        self.dtype = dtype
        self.build_gaps = build_gaps
        
        # Download and load model
        print(f"Downloading model weights to: {model_path}")
        snapshot_download(repo_id="wgsxm/PartCrafter-Scene", local_dir=model_path)
        
        print("Loading PartCrafter model...")
        self.pipeline = PartCrafterPipeline.from_pretrained(model_path).to(device, dtype)
        print("Model loading completed!")
        
        # Setup GAPS if requested
        if self.build_gaps:
            setup_gaps_tools()
    
    def load_test_config(self, config_path: str) -> List[Dict]:
        """Load test configuration"""
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs
    
    def load_gt_mesh(self, mesh_path: str) -> trimesh.Scene:
        """Load Ground Truth mesh"""
        mesh = trimesh.load(mesh_path, process=False)
        return mesh
    
    def run_inference(self, 
                     image_path: str, 
                     num_parts: int,
                     seed: int = 0,
                     num_tokens: int = 2048,
                     num_inference_steps: int = 50,
                     guidance_scale: float = 7.0) -> Tuple[List[trimesh.Trimesh], Image.Image]:
        """Run PartCrafter inference"""
        img_pil = Image.open(image_path)
        
        start_time = time.time()
        outputs = self.pipeline(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=int(1e9),
            use_flash_decoder=False,
        ).meshes
        
        end_time = time.time()
        print(f"Inference time: {end_time - start_time:.2f} seconds")
        
        # Handle None outputs
        for i in range(len(outputs)):
            if outputs[i] is None:
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        
        return outputs, img_pil
    
    def evaluate_single_case(self, 
                           config: Dict,
                           output_dir: str,
                           num_samples: int = 10000) -> Dict[str, Any]:
        """Evaluate a single test case"""
        case_name = Path(config['mesh_path']).stem
        print(f"\nEvaluating case: {case_name}")
        
        try:
            # Load GT mesh
            gt_mesh = self.load_gt_mesh(config['mesh_path'])
            
            # Run PartCrafter inference
            pred_meshes, input_image = self.run_inference(
                image_path=config['image_path'],
                num_parts=config['num_parts'],
                seed=0
            )
            
            # Compute PartCrafter metrics with alignment
            metrics = compute_aligned_metrics(gt_mesh, pred_meshes, num_samples)
            
            # Render comparison
            aligned_gt_scene = metrics.get('aligned_gt_scene')
            aligned_pred_scene = metrics.get('aligned_pred_scene')
            comparison_path = render_comparison_with_alignment(
                gt_mesh, pred_meshes, input_image, output_dir, case_name,
                aligned_gt_scene, aligned_pred_scene
            )
            
            # Save PartCrafter meshes as GLB files
            mesh_dir = save_meshes_with_alignment(
                pred_meshes, gt_mesh, output_dir, case_name,
                aligned_gt_scene, aligned_pred_scene
            )
            
            result = {
                'case_name': case_name,
                'num_parts': config['num_parts'],
                'metrics': metrics,
                'comparison_image': comparison_path,
                'mesh_dir': mesh_dir,
                'success': True
            }
            
            print_case_results(case_name, metrics)
            return result
            
        except Exception as e:
            print(f"Error evaluating case {case_name}: {e}")
            return {
                'case_name': case_name,
                'num_parts': config.get('num_parts', 0),
                'metrics': {
                    'chamfer_distance': float('inf'),
                    'f_score': 0.0,
                    'iou': 0.0,
                    'scene_iou': 0.0
                },
                'error': str(e),
                'success': False
            }
    
    def evaluate_dataset(self, 
                        config_path: str,
                        output_dir: str,
                        num_samples: int = 10000) -> Dict[str, Any]:
        """Evaluate the entire dataset"""
        print(f"Starting dataset evaluation: {config_path}")
        
        # Load test configuration
        configs = self.load_test_config(config_path)
        print(f"Found {len(configs)} test cases")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each case
        results = []
        for config in tqdm(configs, desc="Evaluation progress"):
            result = self.evaluate_single_case(config, output_dir, num_samples)
            results.append(result)
        
        # Calculate overall statistics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            metrics_summary = {
                'chamfer_distance': {
                    'mean': np.mean([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'std': np.std([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'min': np.min([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'max': np.max([r['metrics']['chamfer_distance'] for r in successful_results])
                },
                'f_score': {
                    'mean': np.mean([r['metrics']['f_score'] for r in successful_results]),
                    'std': np.std([r['metrics']['f_score'] for r in successful_results]),
                    'min': np.min([r['metrics']['f_score'] for r in successful_results]),
                    'max': np.max([r['metrics']['f_score'] for r in successful_results])
                },
                'iou': {
                    'mean': np.mean([r['metrics']['iou'] for r in successful_results]),
                    'std': np.std([r['metrics']['iou'] for r in successful_results]),
                    'min': np.min([r['metrics']['iou'] for r in successful_results]),
                    'max': np.max([r['metrics']['iou'] for r in successful_results])
                },
                'scene_iou': {
                    'mean': np.mean([r['metrics']['scene_iou'] for r in successful_results]),
                    'std': np.std([r['metrics']['scene_iou'] for r in successful_results]),
                    'min': np.min([r['metrics']['scene_iou'] for r in successful_results]),
                    'max': np.max([r['metrics']['scene_iou'] for r in successful_results])
                },
                'per_object_metrics': {
                    'all_cds': [cd for r in successful_results for cd in r['metrics'].get('per_object_cds', [])],
                    'all_fscores': [fscore for r in successful_results for fscore in r['metrics'].get('per_object_fscores', [])],
                    'all_ious': [iou for r in successful_results for iou in r['metrics'].get('per_object_ious', [])]
                }
            }
        else:
            metrics_summary = {}
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'config_path': config_path,
                'total_cases': len(configs),
                'successful_cases': len(successful_results),
                'failed_cases': len(results) - len(successful_results),
                'metrics_summary': metrics_summary,
                'detailed_results': results
            }, f, indent=2)
        
        # Create results table
        if successful_results:
            df_data = []
            for result in successful_results:
                df_data.append({
                    'case_name': result['case_name'],
                    'num_parts': result['num_parts'],
                    'chamfer_distance': result['metrics']['chamfer_distance'],
                    'chamfer_distance_std': result['metrics'].get('chamfer_distance_std', 0.0),
                    'f_score': result['metrics']['f_score'],
                    'f_score_std': result['metrics'].get('f_score_std', 0.0),
                    'iou': result['metrics']['iou'],
                    'iou_std': result['metrics'].get('iou_std', 0.0),
                    'scene_iou': result['metrics']['scene_iou'],
                    'num_gt_objects': len(result['metrics'].get('per_object_cds', [])),
                    'num_pred_objects': result['num_parts']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
            
            # Create visualization charts
            create_evaluation_visualizations(df, output_dir)
        
        # Print summary
        print_evaluation_summary(metrics_summary, len(configs), len(successful_results))
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'results': results,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(description='PartCrafter large-scale evaluation script')
    parser.add_argument('--config_path', type=str, 
                       default='data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json',
                       help='Test configuration file path')
    parser.add_argument('--output_dir', type=str, 
                       default='results/evaluation_messy_kitchen',
                       help='Output directory')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter-Scene',
                       help='Model weights path')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of evaluation sample points')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--build_gaps', action='store_true', default=True,
                       help='Setup GAPS tools for evaluation (default: True, assumes GAPS is pre-installed)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create evaluator
    evaluator = PartCrafterEvaluator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float16,
        build_gaps=args.build_gaps
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        config_path=args.config_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

