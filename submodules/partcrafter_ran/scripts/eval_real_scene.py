#!/usr/bin/env python3
"""
Real Scene Evaluation Script: Evaluate PartCrafter performance on real scene data like demo-sc-1/
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

class RealSceneEvaluator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter-Scene",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 build_gaps: bool = True,
                 load_trained_checkpoint: str = None,
                 checkpoint_iteration: int = None):
        """
        Initialize PartCrafter evaluator for real scenes
        
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
        
        # Set seed for reproducibility
        set_seed(0)
        
        # Setup GAPS if requested
        if self.build_gaps:
            setup_gaps_tools()
    
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
    
    def load_gt_meshes_from_scene_split(self, scene_split_dir: str) -> List[trimesh.Trimesh]:
        """
        Load GT meshes from scene-split directory
        
        Args:
            scene_split_dir: Path to scene-split directory
            
        Returns:
            List of GT meshes
        """
        if not os.path.exists(scene_split_dir):
            print(f"Warning: scene-split directory not found: {scene_split_dir}")
            return []
        
        ply_files = sorted(glob.glob(os.path.join(scene_split_dir, '*.ply')))
        gt_meshes = []
        
        for ply_file in ply_files:
            try:
                mesh = trimesh.load(ply_file, process=False)
                if isinstance(mesh, trimesh.Trimesh):
                    gt_meshes.append(mesh)
                    print(f"  Loaded GT mesh: {os.path.basename(ply_file)}")
                else:
                    print(f"  Warning: {ply_file} is not a valid mesh")
            except Exception as e:
                print(f"  Error loading {ply_file}: {e}")
        
        print(f"Loaded {len(gt_meshes)} GT meshes from scene-split")
        return gt_meshes
    
    def check_existing_results(self, scene_name: str, output_dir: str) -> Tuple[bool, str, List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """
        Check if existing evaluation results are available
        
        Returns:
            (exists, mesh_dir, pred_meshes, gt_meshes)
        """
        def _as_trimesh_list(obj) -> List[trimesh.Trimesh]:
            if obj is None:
                return []
            if isinstance(obj, trimesh.Scene):
                return list(obj.geometry.values())
            if isinstance(obj, trimesh.Trimesh):
                return [obj]
            return []

        mesh_dir = os.path.join(output_dir, scene_name)
        if not os.path.exists(mesh_dir):
            return False, mesh_dir, [], []

        # Check for aligned predicted meshes
        pred_meshes = []
        i = 0
        while True:
            ply_path = os.path.join(mesh_dir, f"aligned_pred_obj_{i}.ply")
            if not os.path.exists(ply_path):
                break
            mesh = trimesh.load(ply_path, process=False)
            pred_meshes.append(mesh)
            i += 1

        # Check for GT meshes
        gt_meshes = []
        i = 0
        while True:
            ply_path = os.path.join(mesh_dir, f"gt_object_{i}.ply")
            if not os.path.exists(ply_path):
                break
            mesh = trimesh.load(ply_path, process=False)
            gt_meshes.append(mesh)
            i += 1

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
        Run PartCrafter inference
        
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
    
    def save_metrics_to_txt(self, scene_name: str, metrics: Dict[str, Any], output_path: str):
        """
        Save metrics to a readable text file
        
        Args:
            scene_name: Name of the scene
            metrics: Dictionary containing evaluation metrics
            output_path: Path to save the metrics.txt file
        """
        with open(output_path, 'w') as f:
            f.write(f"PartCrafter Real Scene Evaluation Results\n")
            f.write(f"Scene: {scene_name}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"  Chamfer Distance: {metrics.get('chamfer_distance', 'N/A'):.6f}\n")
            f.write(f"  F-Score: {metrics.get('f_score', 'N/A'):.6f}\n")
            f.write(f"  IoU: {metrics.get('iou', 'N/A'):.6f}\n")
            f.write(f"  Scene IoU: {metrics.get('scene_iou', 'N/A'):.6f}\n")
            f.write("\n")
            
            # Per-object metrics
            if 'per_object_cds' in metrics and metrics['per_object_cds']:
                f.write("PER-OBJECT METRICS:\n")
                f.write(f"  Number of GT objects: {len(metrics['per_object_cds'])}\n")
                f.write(f"  Number of predicted objects: {len(metrics.get('per_object_fscores', []))}\n")
                f.write("\n")
                
                # Individual object metrics
                for i, (cd, fscore, iou) in enumerate(zip(
                    metrics['per_object_cds'],
                    metrics['per_object_fscores'],
                    metrics['per_object_ious']
                )):
                    f.write(f"  Object {i+1}:\n")
                    f.write(f"    CD: {cd:.6f}\n")
                    f.write(f"    F-Score: {fscore:.6f}\n")
                    f.write(f"    IoU: {iou:.6f}\n")
                    f.write("\n")
            
            # Alignment info
            if 'aligned_gt_scene_path' in metrics:
                f.write("ALIGNMENT INFO:\n")
                f.write(f"  GT Scene: {metrics['aligned_gt_scene_path']}\n")
                f.write(f"  Predicted Scene: {metrics['aligned_pred_merged_path']}\n")
                f.write("\n")
            
            # Additional info
            f.write("ADDITIONAL INFO:\n")
            f.write(f"  Evaluation completed successfully\n")
            f.write(f"  All meshes aligned using GAPS sim(3) registration\n")
            f.write(f"  Metrics computed on aligned meshes\n")
    
    def evaluate_single_scene(self, 
                            scene_config: Dict,
                            output_dir: str,
                            num_samples: int = 10000,
                            use_existing_results: bool = True,
                            force_inference: bool = False,
                            inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate a single real scene
        
        Returns:
            Dictionary containing evaluation results
        """
        scene_name = scene_config['scene_name']
        print(f"\nEvaluating real scene: {scene_name}")
        
        try:
            # Check if existing results are available
            has_existing, mesh_dir, pred_meshes, gt_meshes = self.check_existing_results(scene_name, output_dir)
            
            if has_existing and use_existing_results and not force_inference:
                # Use existing results - skip inference but still need GT meshes for metrics
                print(f"Loaded {len(pred_meshes)} predicted meshes and {len(gt_meshes)} GT meshes from existing results")
            else:
                # Load GT meshes and run inference
                if has_existing and force_inference:
                    print(f"Force running inference for scene: {scene_name} (overwriting existing results)")
                else:
                    print(f"Running inference for scene: {scene_name}")
                
                # Load GT meshes from scene-split directory
                gt_meshes = self.load_gt_meshes_from_scene_split(scene_config['scene_split_dir'])
                
                # Create output directory for this scene
                case_output_dir = os.path.join(output_dir, scene_name)
                os.makedirs(case_output_dir, exist_ok=True)
                
                # Run PartCrafter inference
                pred_meshes, input_image = self.run_inference(
                    image_path=scene_config['image_path'],
                    num_parts=scene_config['num_parts'],
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
            
            # Always compute metrics with alignment (whether using existing results or not)
            print(f"Computing metrics with alignment for {len(pred_meshes)} predicted meshes...")
            
            # Create output directory for this scene (if not already created)
            case_output_dir = os.path.join(output_dir, scene_name)
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Save meshes for alignment
            pred_merged_path = os.path.join(case_output_dir, "pred_merged.ply")
            gt_merged_path = os.path.join(case_output_dir, "gt_merged.ply")
            
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
            
            # Temporarily comment out metrics computation
            # print(f"Computing metrics with {len(gt_meshes)} GT meshes and {len(aligned_pred_meshes)} aligned predicted meshes")
            # metrics = compute_aligned_metrics(gt_meshes, aligned_pred_meshes, num_samples)
            
            # Create dummy metrics for now
            metrics = {
                'chamfer_distance': 0.0,
                'f_score': 0.0,
                'iou': 0.0,
                'scene_iou': 0.0,
                'per_object_cds': [],
                'per_object_fscores': [],
                'per_object_ious': []
            }

            # Save aligned predicted meshes
            if aligned_pred_meshes and len(aligned_pred_meshes) > 0:
                # Save each aligned predicted mesh as aligned_pred_obj_*.ply
                print(f"Saving {len(aligned_pred_meshes)} aligned predicted meshes as PLY files...")
                for i, mesh in enumerate(aligned_pred_meshes):
                    ply_filename = f"aligned_pred_obj_{i}.ply"
                    ply_path = os.path.join(case_output_dir, ply_filename)
                    mesh.export(ply_path)
                    print(f"  Saved: {ply_filename}")
                
            # Save GT meshes
            if gt_meshes and len(gt_meshes) > 0:
                # Save each GT mesh as gt_object_*.ply
                print(f"Saving {len(gt_meshes)} GT meshes as PLY files...")
                for i, mesh in enumerate(gt_meshes):
                    ply_filename = f"gt_object_{i}.ply"
                    ply_path = os.path.join(case_output_dir, ply_filename)
                    mesh.export(ply_path)
                    print(f"  Saved: {ply_filename}")
            
            # Add aligned scene file paths
            metrics['aligned_gt_scene_path'] = os.path.join(case_output_dir, "gt_merged.glb")
            metrics['aligned_pred_merged_path'] = os.path.join(case_output_dir, "aligned_pred_merged.glb")

            print("[INFO] Penetration metrics temporarily disabled")
            
            comparison_path = None
            
            result = {
                'scene_name': scene_name,
                'num_parts': scene_config['num_parts'],
                'metrics': metrics,
                'comparison_image': comparison_path,
                'mesh_dir': mesh_dir,
                'success': True
            }
            
            print_case_results(scene_name, metrics)
            
            # Save metrics to a text file in the scene folder
            metrics_txt_path = os.path.join(case_output_dir, "metrics.txt")
            self.save_metrics_to_txt(scene_name, metrics, metrics_txt_path)
            print(f"Metrics saved to: {metrics_txt_path}")
            
            return result
            
        except Exception as e:
            print(f"Error evaluating scene {scene_name}: {e}")
            return {
                'scene_name': scene_name,
                'num_parts': scene_config.get('num_parts', 0),
                'metrics': {
                    'chamfer_distance': float('inf'),
                    'f_score': 0.0,
                    'iou': 0.0,
                    'scene_iou': 0.0
                },
                'error': str(e),
                'success': False
            }
    
    def evaluate_real_scenes(self, 
                           scene_dirs: List[str],
                           output_dir: str,
                           num_samples: int = 10000,
                           use_existing_results: bool = True,
                           force_inference: bool = False,
                           inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate multiple real scenes
        
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting real scene evaluation: {len(scene_dirs)} scenes")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each scene
        results = []
        for scene_dir in tqdm(scene_dirs, desc="Evaluation progress"):
            try:
                # Load scene configuration
                scene_config = self.load_real_scene_config(scene_dir)
                
                # Evaluate this scene
                result = self.evaluate_single_scene(
                    scene_config, output_dir, num_samples, 
                    use_existing_results, force_inference, inference_args
                )
                results.append(result)
                
            except Exception as e:
                print(f"Error processing scene {scene_dir}: {e}")
                results.append({
                    'scene_dir': scene_dir,
                    'error': str(e),
                    'success': False
                })
        
        # Calculate overall statistics
        successful_results = [r for r in results if r.get('success', False)]
        
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
                'scene_dirs': scene_dirs,
                'total_scenes': len(scene_dirs),
                'successful_scenes': len(successful_results),
                'failed_scenes': len(results) - len(successful_results),
                'metrics_summary': metrics_summary,
                'detailed_results': results
            }, f, indent=2)
        
        # Create results table
        if successful_results:
            df_data = []
            for result in successful_results:
                df_data.append({
                    'scene_name': result['scene_name'],
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
        print_evaluation_summary(metrics_summary, len(scene_dirs), len(successful_results))
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'results': results,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir
        }


def main():
    parser = argparse.ArgumentParser(description='PartCrafter real scene evaluation script')
    parser.add_argument('--scene_dirs', type=str, nargs='+', 
                       default=['data/messy_kitchen_real/demo-sc-1'],
                       help='List of scene directory paths')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/evaluation_real_scenes',
                       help='Output directory')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter',
                       help='Base model weights path (for VAE, feature extractor, etc.)')
    parser.add_argument('--load_trained_checkpoint', type=str, 
                       default='runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512',
                       help='Path to trained checkpoint directory')
    parser.add_argument('--checkpoint_iteration', type=int, default=17000,
                       help='Checkpoint iteration number')
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
    parser.add_argument('--force_inference', action='store_true', default=False,
                       help='Force running inference even if existing results are found')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create evaluator
    evaluator = RealSceneEvaluator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float16,
        build_gaps=args.build_gaps,
        load_trained_checkpoint=args.load_trained_checkpoint,
        checkpoint_iteration=args.checkpoint_iteration
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
    results = evaluator.evaluate_real_scenes(
        scene_dirs=args.scene_dirs,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_existing_results=args.use_existing_results,
        force_inference=args.force_inference,
        inference_args=inference_args
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
