#!/usr/bin/env python3
"""
Large-scale test script: Evaluate PartCrafter performance on messy kitchen dataset
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
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import yaml

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.utils.eval_utils import (
    setup_gaps_tools, compute_aligned_metrics, 
    save_meshes_with_alignment, align_merged_meshes_with_gaps_and_get_transform,
    apply_gaps_transformation_to_meshes
)
from src.utils.visualization_utils import (
    render_comparison_with_alignment, create_evaluation_visualizations,
    print_evaluation_summary, print_case_results
)
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

from huggingface_hub import snapshot_download
from accelerate.utils import set_seed

MAX_NUM_PARTS = 16

class PartCrafterEvaluator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter-Scene",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 build_gaps: bool = True):
        """
        Initialize PartCrafter evaluator
        
        Args:
            model_path: Model weights path
            device: Computing device
            dtype: Data type
            build_gaps: Whether to build GAPS tools for evaluation
        """
        self.device = device
        self.dtype = dtype
        self.build_gaps = build_gaps
        
        # Download and load model - exactly match inference_partcrafter.py
        print(f"Downloading model weights to: {model_path}")
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=model_path)
        
        # Download RMBG weights - exactly match inference_partcrafter.py
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        
        # init rmbg model for background removal - exactly match inference_partcrafter.py
        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        self.rmbg_net.eval()
        
        print("Loading PartCrafter model...")
        self.pipeline = PartCrafterPipeline.from_pretrained(model_path).to(device, dtype)
        print("Model loading completed!")
        
        # Set seed for reproducibility - same as inference_partcrafter.py
        set_seed(0)
        
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
    
# from typing import Tuple, List
# import os
# import trimesh

    def check_existing_results(
        self, case_name: str, output_dir: str
    ) -> Tuple[bool, str, List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """
        Returns: (exists, mesh_dir, pred_meshes, gt_meshes)
        """

        def _as_trimesh_list(obj) -> List[trimesh.Trimesh]:
            if obj is None:
                return []
            if isinstance(obj, trimesh.Scene):
                return list(obj.geometry.values())
            if isinstance(obj, trimesh.Trimesh):
                return [obj]
            return []

        def _load_parts(dir_path: str, prefix: str) -> List[trimesh.Trimesh]:
            meshes: List[trimesh.Trimesh] = []
            i = 0
            while True:
                p = os.path.join(dir_path, f"{prefix}_{i:02d}.glb")
                if not os.path.exists(p):
                    break
                m = trimesh.load(p, process=False)
                meshes.extend(_as_trimesh_list(m))
                i += 1
            return meshes

        mesh_dir = os.path.join(output_dir, case_name)
        if not os.path.exists(mesh_dir):
            return False, mesh_dir, [], []

        pred_meshes = _load_parts(mesh_dir, "pred_part")
        gt_meshes   = _load_parts(mesh_dir, "gt_part")

        if not pred_meshes:
            pred_merged_path = os.path.join(mesh_dir, "pred_merged.glb")
            if os.path.exists(pred_merged_path):
                pred_meshes = _as_trimesh_list(trimesh.load(pred_merged_path, process=False))

        if not gt_meshes:
            gt_merged_path = os.path.join(mesh_dir, "gt_merged.glb")
            if os.path.exists(gt_merged_path):
                gt_meshes = _as_trimesh_list(trimesh.load(gt_merged_path, process=False))

        exists = bool(pred_meshes or gt_meshes)
        return exists, mesh_dir, pred_meshes, gt_meshes

    
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
        Run PartCrafter inference - exactly matching inference_partcrafter.py run_triposg function
        
        Returns:
            generated_meshes: List of generated meshes
            processed_image: Processed input image
        """
        # Exactly match inference_partcrafter.py run_triposg function
        assert 1 <= num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
        if rmbg:
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        else:
            img_pil = Image.open(image_path)
        start_time = time.time()
        outputs = self.pipeline(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        ).meshes
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        for i in range(len(outputs)):
            if outputs[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        return outputs, img_pil
    

    

    

    

    
    def evaluate_single_case(self, 
                           config: Dict,
                           output_dir: str,
                           num_samples: int = 10000,
                           use_existing_results: bool = True,
                           force_inference: bool = False,
                           inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate a single test case
        
        Returns:
            Dictionary containing evaluation results
        """
        case_name = Path(config['mesh_path']).stem
        print(f"\nEvaluating case: {case_name}")
        
        try:
            # Check if existing results are available
            has_existing, mesh_dir, pred_meshes, gt_meshes = self.check_existing_results(case_name, output_dir)
            
            if has_existing and use_existing_results and not force_inference:
                # Use existing results - skip inference but still need GT mesh for metrics
                # Load GT mesh from original path for metrics computation
                gt_mesh = self.load_gt_mesh(config['mesh_path'])
                print(f"Loaded {len(pred_meshes)} predicted meshes and {len(gt_meshes)} GT meshes from existing results")
            else:
                # Load GT mesh and run inference
                if has_existing and force_inference:
                    print(f"Force running inference for case: {case_name} (overwriting existing results)")
                else:
                    print(f"Running inference for case: {case_name}")
                
                gt_mesh = self.load_gt_mesh(config['mesh_path'])
                
                # Create output directory for this case
                case_output_dir = os.path.join(output_dir, case_name)
                os.makedirs(case_output_dir, exist_ok=True)
                
                # Run PartCrafter inference with exact same parameters as inference_partcrafter.py
                pred_meshes, input_image = self.run_inference(
                    image_path=config['image_path'],
                    num_parts=config['num_parts'],
                    seed=inference_args.get('seed', 0) if inference_args else 0,
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
                pred_meshes[0].export(os.path.join(case_output_dir, "pred_part_00.glb"))
            
            # Always compute metrics with alignment (whether using existing results or not)
            print(f"Computing metrics with alignment for {len(pred_meshes)} predicted meshes...")
            
            # Extract GT meshes from gt_mesh if not already loaded
            if not gt_meshes:
                if isinstance(gt_mesh, trimesh.Scene):
                    gt_meshes = list(gt_mesh.geometry.values())
                elif isinstance(gt_mesh, trimesh.Trimesh):
                    gt_meshes = [gt_mesh]
                else:
                    gt_meshes = []
            
            # Create output directory for this case (if not already created)
            case_output_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Save meshes for alignment
            pred_merged_path = os.path.join(case_output_dir, "pred_merged.ply")
            gt_merged_path = os.path.join(case_output_dir, "gt_merged.ply")
            aligned_pred_path = os.path.join(case_output_dir, "aligned_pred_merged.ply")
            
            # Merge and save meshes
            assert pred_meshes is not None
            assert gt_meshes is not None
            pred_merged = trimesh.util.concatenate(pred_meshes)
            pred_merged.export(pred_merged_path)
       
            gt_merged = trimesh.util.concatenate(gt_meshes)
            gt_merged.export(gt_merged_path)
    
            # Run GAPS alignment if both meshes exist
            print("Running GAPS alignment with transformation extraction...")
                
            # Get aligned merged mesh, transformation matrix, and scale factor
            gt_merged_aligned, aligned_pred_merged, transformation_matrix, scale_factor = align_merged_meshes_with_gaps_and_get_transform(gt_merged, pred_merged)
                    
            aligned_pred_merged.export(os.path.join(case_output_dir, "aligned_pred_merged.glb"))
            gt_merged.export(os.path.join(case_output_dir, "gt_merged.glb"))

            print(f"GAPS alignment successful")
            print(f"Transformation matrix shape: {transformation_matrix.shape}")
            print(f"Scale factor: {scale_factor}")
            print(f"Transformation matrix:\n{transformation_matrix}")

            assert np.allclose(transformation_matrix, np.eye(4)) != True
            assert scale_factor != 1.0 
            
            # Apply the transformation to individual predicted meshes
            print(f"Applying transformation to {len(pred_meshes)} individual predicted meshes...")
            aligned_pred_meshes = apply_gaps_transformation_to_meshes(pred_meshes, transformation_matrix)
            print(f"Successfully applied transformation to {len(aligned_pred_meshes)} individual predicted meshes")
            
            # Save transformation info for debugging
            transformation_info = {
                'transformation_matrix': transformation_matrix.tolist(),
                'scale_factor': scale_factor,
                'matrix_shape': transformation_matrix.shape,
                'is_identity': bool(np.allclose(transformation_matrix, np.eye(4)))
            }
                    
            # Save transformation info to file
            transformation_path = os.path.join(case_output_dir, "transformation_info.json")
            with open(transformation_path, 'w') as f:
                json.dump(transformation_info, f, indent=2)
            print(f"Transformation info saved to: {transformation_path}")
       
            aligned_gt_scene = trimesh.Scene(gt_meshes)
            aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
            
            print(f"Computing metrics with {len(gt_meshes)} GT meshes and {len(aligned_pred_meshes)} aligned predicted meshes")
            metrics = compute_aligned_metrics(gt_meshes, aligned_pred_meshes, num_samples)

            # Save some debug meshes
            if aligned_pred_meshes and len(aligned_pred_meshes) > 0:
                aligned_pred_meshes[0].export(os.path.join(case_output_dir, "pred_part_00.glb"))
            if gt_merged is not None:
                gt_merged.export(os.path.join(case_output_dir, "gt_merged.glb"))
            
            # Add aligned scenes to metrics for visualization
            metrics['aligned_gt_scene'] = aligned_gt_scene
            metrics['aligned_pred_scene'] = aligned_pred_scene
            
            comparison_path = None
            
            # Save alignment results if not using existing results or if alignment results don't exist
            if not has_existing or not os.path.exists(os.path.join(mesh_dir, "gt_merged_aligned.glb")):
                mesh_dir = save_meshes_with_alignment(
                    pred_meshes, gt_mesh, output_dir, case_name,
                    aligned_gt_scene, aligned_pred_scene, metrics
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
                        num_samples: int = 10000,
                        use_existing_results: bool = True,
                        force_inference: bool = False,
                        inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate the entire dataset
        
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting dataset evaluation: {config_path}")
        
        # Load test configuration
        configs = self.load_test_config(config_path)
        print(f"Found {len(configs)} test cases")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each case
        results = []
        for config in tqdm(configs, desc="Evaluation progress"):
            result = self.evaluate_single_case(
                config, output_dir, num_samples, 
                use_existing_results, force_inference, inference_args
            )
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
    


def save_meshes_standalone(pred_meshes, gt_mesh, output_path, case_name="test_case"):
    """
    Standalone function to save meshes as GLB files
    
    Args:
        pred_meshes: List of predicted trimesh.Trimesh objects
        gt_mesh: Ground Truth trimesh.Scene or trimesh.Trimesh object
        output_path: Directory to save the GLB files
        case_name: Name for the case (used in file naming)
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save individual predicted meshes
    for i, mesh in enumerate(pred_meshes):
        mesh.export(os.path.join(output_path, f"pred_part_{i:02d}.glb"))
    
    # Save merged predicted mesh
    if pred_meshes:
        merged_mesh = trimesh.util.concatenate(pred_meshes)
        merged_mesh.export(os.path.join(output_path, "pred_merged.glb"))
    
    # Save GT mesh
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh.export(os.path.join(output_path, "gt_merged.glb"))
    elif isinstance(gt_mesh, trimesh.Trimesh):
        gt_mesh.export(os.path.join(output_path, "gt_merged.glb"))
    
    print(f"All meshes saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='PartCrafter large-scale evaluation script')
    parser.add_argument('--config_path', type=str, 
                       default='data/preprocessed_data_scenes_objects_demo_test/objects_demo_configs.json',
                       help='Test configuration file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./results',
                       help='Output directory')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter',
                       help='Model weights path')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of evaluation sample points')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Number of tokens for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0, help='Guidance scale for generation')
    parser.add_argument('--max_num_expanded_coords', type=int, default=1e9, help='Maximum number of expanded coordinates')
    parser.add_argument('--use_flash_decoder', action='store_true', help='Use flash decoder')
    parser.add_argument('--rmbg', action='store_true', help='Use background removal')
    parser.add_argument('--build_gaps', action='store_true', default=True,
                       help='Setup GAPS tools for evaluation (default: True, assumes GAPS is pre-installed)')
    parser.add_argument('--use_existing_results', action='store_true', default=True,
                       help='Use existing prediction results if available (default: True)')
    parser.add_argument('--force_inference', action='store_true', default=True,
                       help='Force running inference even if existing results are found')

    
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
    
    # Prepare inference arguments
    inference_args = {
        'seed': args.seed,
        'num_tokens': args.num_tokens,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'max_num_expanded_coords': args.max_num_expanded_coords,
        'use_flash_decoder': args.use_flash_decoder,
        'rmbg': args.rmbg
    }
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        config_path=args.config_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_existing_results=args.use_existing_results,
        force_inference=args.force_inference,
        inference_args=inference_args
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
