#!/usr/bin/env python3
"""
Real scene penetration score calculation script for messy kitchen scenarios.

This script processes real scene data from data/messy_kitchen_real/demo-sc-1_aligned_pred
and calculates penetration scores between aligned predicted meshes and ground truth meshes.

Features:
- High-resolution TSDF (80x80x80 = 512K voxels) for accurate penetration calculation
- Real scene data processing with aligned predicted meshes
- Whole-scene Sim(3) transformation: normalizes entire scene to [0, 0.3] range as a unit
- Preserves all relative positions and spatial relationships between objects
- Uses combined bounds of all meshes to compute single transformation for entire scene
- Comprehensive penetration analysis and visualization
- Saves normalized meshes for inspection
"""

import os
import json
from pathlib import Path
import numpy as np
import trimesh
import sys

# Add the src directory to the path to import utils_giga
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Try to import utils_giga functions, but handle open3d dependency gracefully
UTILS_GIGA_AVAILABLE = False
try:
    from utils_giga import (
        compute_multi_object_completion_metrics_messy_kitchen,
        compute_chamfer_and_iou_messy_kitchen,
        compute_penetration_level,
        compute_penetration_level_detailed,
        mesh_to_tsdf_messy_kitchen,
        compute_bounding_box_iou
    )
    UTILS_GIGA_AVAILABLE = True
    print("✓ Successfully imported utils_giga functions")
except ImportError as e:
    print(f"⚠ Warning: Could not import utils_giga functions: {e}")
    print("  This script will only perform mesh normalization and saving.")
    UTILS_GIGA_AVAILABLE = False

def compute_sim3_transformation(mesh, target_size=0.3):
    """
    Compute Sim(3) transformation parameters to normalize mesh to [0, target_size] range
    
    Args:
        mesh: Reference mesh to compute transformation from
        target_size: Target scene size (default: 0.3)
        
    Returns:
        tuple: (translation_vector, scale_factor, final_translation_vector)
    """
    # Get mesh bounds
    bounds = mesh.bounds
    min_coords = bounds[0]
    max_coords = bounds[1]
    dimensions = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    
    # Step 1: Translation to center at origin
    translation_to_origin = -center
    
    # Step 2: Scale to fit within [0, target_size]
    max_dimension = np.max(dimensions)
    if max_dimension > 0:
        scale_factor = target_size / max_dimension
    else:
        scale_factor = 1.0
    
    # Step 3: Final translation to position in [0, target_size] range
    final_translation = np.array([target_size/2, target_size/2, target_size/2])
    
    return translation_to_origin, scale_factor, final_translation

def apply_sim3_transformation(mesh, translation_to_origin, scale_factor, final_translation):
    """
    Apply Sim(3) transformation to a mesh using given parameters
    
    Args:
        mesh: Input mesh
        translation_to_origin: Translation vector to center at origin
        scale_factor: Scaling factor
        final_translation: Final translation vector
        
    Returns:
        Transformed mesh
    """
    # Make a copy
    transformed_mesh = mesh.copy()
    
    # Apply transformations in order
    transformed_mesh.apply_translation(translation_to_origin)
    transformed_mesh.apply_scale(scale_factor)
    transformed_mesh.apply_translation(final_translation)
    
    return transformed_mesh

def load_real_scene_meshes(scene_dir):
    """
    Load aligned predicted meshes and ground truth meshes from a real scene directory
    
    Args:
        scene_dir (Path): Path to scene directory
        
    Returns:
        tuple: (aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh)
    """
    aligned_pred_meshes = []
    gt_meshes = []
    merged_aligned_pred_mesh = None
    merged_gt_mesh = None
    
    try:
        print(f"Loading meshes from: {scene_dir}")
        
        # Load aligned predicted meshes (aligned_pred_obj_*.ply)
        aligned_pred_files = sorted(scene_dir.glob("aligned_pred_obj_*.ply"))
        for pred_file in aligned_pred_files:
            try:
                mesh = trimesh.load_mesh(str(pred_file))
                aligned_pred_meshes.append(mesh)
                print(f"  Loaded aligned predicted mesh: {pred_file.name}")
                print(f"    - Vertices: {len(mesh.vertices)}")
                print(f"    - Faces: {len(mesh.faces)}")
                print(f"    - Bounds: {mesh.bounds}")
                print(f"    - Is watertight: {mesh.is_watertight}")
            except Exception as e:
                print(f"  Error loading {pred_file.name}: {e}")
        
        # Load ground truth meshes (gt_object_*.ply)
        gt_files = sorted(scene_dir.glob("gt_object_*.ply"))
        for gt_file in gt_files:
            try:
                mesh = trimesh.load_mesh(str(gt_file))
                gt_meshes.append(mesh)
                print(f"  Loaded GT mesh: {gt_file.name}")
                print(f"    - Vertices: {len(mesh.vertices)}")
                print(f"    - Faces: {len(mesh.faces)}")
                print(f"    - Bounds: {mesh.bounds}")
                print(f"    - Is watertight: {mesh.is_watertight}")
            except Exception as e:
                print(f"  Error loading {gt_file.name}: {e}")
        
        # Load merged aligned predicted mesh (aligned_pred_merged.glb or .ply)
        pred_merged_files = [
            scene_dir / "aligned_pred_merged.glb",
            scene_dir / "aligned_pred_merged.ply"
        ]
        for pred_merged_file in pred_merged_files:
            if pred_merged_file.exists():
                try:
                    mesh = trimesh.load_mesh(str(pred_merged_file))
                    merged_aligned_pred_mesh = mesh
                    print(f"  Loaded merged aligned predicted mesh: {pred_merged_file.name}")
                    print(f"    - Vertices: {len(mesh.vertices)}")
                    print(f"    - Faces: {len(mesh.faces)}")
                    print(f"    - Bounds: {mesh.bounds}")
                    break
                except Exception as e:
                    print(f"  Error loading {pred_merged_file.name}: {e}")
        
        # Load merged ground truth mesh (gt_merged.glb or .ply)
        gt_merged_files = [
            scene_dir / "gt_merged.glb",
            scene_dir / "gt_merged.ply"
        ]
        for gt_merged_file in gt_merged_files:
            if gt_merged_file.exists():
                try:
                    mesh = trimesh.load_mesh(str(gt_merged_file))
                    merged_gt_mesh = mesh
                    print(f"  Loaded merged GT mesh: {gt_merged_file.name}")
                    print(f"    - Vertices: {len(mesh.vertices)}")
                    print(f"    - Faces: {len(mesh.faces)}")
                    print(f"    - Bounds: {mesh.bounds}")
                    break
                except Exception as e:
                    print(f"  Error loading {gt_merged_file.name}: {e}")
        
        # If no merged meshes found, create them from individual meshes
        if merged_aligned_pred_mesh is None and aligned_pred_meshes:
            print("  Creating merged aligned predicted mesh from individual meshes...")
            merged_aligned_pred_mesh = trimesh.util.concatenate(aligned_pred_meshes)
            print(f"    - Merged vertices: {len(merged_aligned_pred_mesh.vertices)}")
            print(f"    - Merged faces: {len(merged_aligned_pred_mesh.faces)}")
        
        if merged_gt_mesh is None and gt_meshes:
            print("  Creating merged GT mesh from individual meshes...")
            merged_gt_mesh = trimesh.util.concatenate(gt_meshes)
            print(f"    - Merged vertices: {len(merged_gt_mesh.vertices)}")
            print(f"    - Merged faces: {len(merged_gt_mesh.faces)}")
        
        # Compute Sim(3) transformation from the combined bounds of all meshes to normalize the entire scene
        print("  Computing Sim(3) transformation for entire scene normalization...")
        
        # Collect all meshes to compute combined scene bounds
        all_meshes = []
        if aligned_pred_meshes:
            all_meshes.extend(aligned_pred_meshes)
        if gt_meshes:
            all_meshes.extend(gt_meshes)
        if merged_aligned_pred_mesh is not None:
            all_meshes.append(merged_aligned_pred_mesh)
        if merged_gt_mesh is not None:
            all_meshes.append(merged_gt_mesh)
        
        if all_meshes:
            # Create a temporary combined mesh to get overall scene bounds
            print("    Computing overall scene bounds from all meshes...")
            combined_mesh = trimesh.util.concatenate(all_meshes)
            original_scene_bounds = combined_mesh.bounds
            print(f"    - Original scene bounds: {original_scene_bounds}")
            
            # Compute transformation parameters based on entire scene
            translation_to_origin, scale_factor, final_translation = compute_sim3_transformation(
                combined_mesh, target_size=0.3
            )
            
            print(f"    - Translation to origin: {translation_to_origin}")
            print(f"    - Scale factor: {scale_factor:.6f}")
            print(f"    - Final translation: {final_translation}")
            
            # Apply the same Sim(3) transformation to all meshes to preserve relative positions
            print("  Applying same Sim(3) transformation to all meshes...")
            
            # Transform merged aligned predicted mesh
            if merged_aligned_pred_mesh is not None:
                print("    Transforming merged aligned predicted mesh...")
                original_bounds = merged_aligned_pred_mesh.bounds
                merged_aligned_pred_mesh = apply_sim3_transformation(
                    merged_aligned_pred_mesh, translation_to_origin, scale_factor, final_translation
                )
                normalized_bounds = merged_aligned_pred_mesh.bounds
                print(f"      - Original bounds: {original_bounds}")
                print(f"      - Transformed bounds: {normalized_bounds}")
            
            # Transform individual aligned predicted meshes
            if aligned_pred_meshes:
                print("    Transforming individual aligned predicted meshes...")
                for i, mesh in enumerate(aligned_pred_meshes):
                    original_bounds = mesh.bounds
                    aligned_pred_meshes[i] = apply_sim3_transformation(
                        mesh, translation_to_origin, scale_factor, final_translation
                    )
                    normalized_bounds = aligned_pred_meshes[i].bounds
                    print(f"      - Mesh {i}: Original bounds: {original_bounds}")
                    print(f"      - Mesh {i}: Transformed bounds: {normalized_bounds}")
            
            # Transform GT meshes (both individual and merged)
            if gt_meshes:
                print("    Transforming individual GT meshes...")
                for i, mesh in enumerate(gt_meshes):
                    original_bounds = mesh.bounds
                    gt_meshes[i] = apply_sim3_transformation(
                        mesh, translation_to_origin, scale_factor, final_translation
                    )
                    normalized_bounds = gt_meshes[i].bounds
                    print(f"      - GT Mesh {i}: Original bounds: {original_bounds}")
                    print(f"      - GT Mesh {i}: Transformed bounds: {normalized_bounds}")
            
            if merged_gt_mesh is not None:
                print("    Transforming merged GT mesh...")
                original_bounds = merged_gt_mesh.bounds
                merged_gt_mesh = apply_sim3_transformation(
                    merged_gt_mesh, translation_to_origin, scale_factor, final_translation
                )
                normalized_bounds = merged_gt_mesh.bounds
                print(f"      - Original bounds: {original_bounds}")
                print(f"      - Transformed bounds: {normalized_bounds}")
            
            # Verify final scene bounds
            final_all_meshes = []
            if aligned_pred_meshes:
                final_all_meshes.extend(aligned_pred_meshes)
            if gt_meshes:
                final_all_meshes.extend(gt_meshes)
            if merged_aligned_pred_mesh is not None:
                final_all_meshes.append(merged_aligned_pred_mesh)
            if merged_gt_mesh is not None:
                final_all_meshes.append(merged_gt_mesh)
            
            if final_all_meshes:
                final_combined_mesh = trimesh.util.concatenate(final_all_meshes)
                final_scene_bounds = final_combined_mesh.bounds
                print(f"    - Final scene bounds after transformation: {final_scene_bounds}")
        else:
            print("    Warning: No meshes found for scene normalization")
        
        # Save normalized meshes to files for inspection
        if merged_aligned_pred_mesh is not None:
            print("  Saving normalized meshes to files...")
            
            # Create normalized_meshes directory
            normalized_dir = scene_dir / "normalized_meshes"
            normalized_dir.mkdir(exist_ok=True)
            
            # Save merged aligned predicted mesh
            merged_pred_path = normalized_dir / "normalized_merged_aligned_pred.ply"
            merged_aligned_pred_mesh.export(str(merged_pred_path))
            print(f"    Saved normalized merged aligned pred mesh: {merged_pred_path}")
            
            # Save individual aligned predicted meshes
            if aligned_pred_meshes:
                for i, mesh in enumerate(aligned_pred_meshes):
                    mesh_path = normalized_dir / f"normalized_aligned_pred_obj_{i}.ply"
                    mesh.export(str(mesh_path))
                    print(f"    Saved normalized aligned pred mesh {i}: {mesh_path}")
            
            # Save individual GT meshes
            if gt_meshes:
                for i, mesh in enumerate(gt_meshes):
                    mesh_path = normalized_dir / f"normalized_gt_obj_{i}.ply"
                    mesh.export(str(mesh_path))
                    print(f"    Saved normalized GT mesh {i}: {mesh_path}")
            
            # Save merged GT mesh
            if merged_gt_mesh is not None:
                merged_gt_path = normalized_dir / "normalized_merged_gt.ply"
                merged_gt_mesh.export(str(merged_gt_path))
                print(f"    Saved normalized merged GT mesh: {merged_gt_path}")
        
        print(f"  Scene {scene_dir.name}: {len(aligned_pred_meshes)} aligned pred meshes, {len(gt_meshes)} GT meshes")
        print(f"    Merged aligned pred mesh: {merged_aligned_pred_mesh is not None}, Merged GT mesh: {merged_gt_mesh is not None}")
        
    except Exception as e:
        print(f"  Error processing scene directory {scene_dir}: {e}")
    
    return aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh

def create_real_scene_visualization(aligned_pred_meshes, gt_meshes, scene_dir, scene_name):
    """
    Create visualization of aligned predicted vs ground truth meshes with color coding
    
    Args:
        aligned_pred_meshes: List of aligned predicted meshes
        gt_meshes: List of ground truth meshes  
        scene_dir: Scene directory path
        scene_name: Name of the scene
    """
    try:
        # Create real_scene_results directory
        results_dir = scene_dir / "real_scene_results"
        results_dir.mkdir(exist_ok=True)
        
        print(f"  Creating real scene visualization in: {results_dir}")
        
        # Create scenes with color coding
        aligned_pred_scene = trimesh.Scene()
        gt_scene = trimesh.Scene()
        comparison_scene = trimesh.Scene()
        
        # Color palette for different objects
        colors = [
            [1.0, 0.0, 0.0, 0.8],  # Red with 80% opacity
            [0.0, 1.0, 0.0, 0.8],  # Green with 80% opacity  
            [0.0, 0.0, 1.0, 0.8],  # Blue with 80% opacity
            [1.0, 1.0, 0.0, 0.8],  # Yellow with 80% opacity
            [1.0, 0.0, 1.0, 0.8],  # Magenta with 80% opacity
            [0.0, 1.0, 1.0, 0.8],  # Cyan with 80% opacity
        ]
        
        # Add aligned predicted meshes to scene
        for i, mesh in enumerate(aligned_pred_meshes):
            if i < len(colors):
                color = colors[i]
            else:
                # Generate a color if we have more objects than predefined colors
                hue = (i * 137.508) % 360  # Golden angle approximation
                color = [hue/360, 0.8, 0.8, 0.8]  # HSV to RGB approximation
            
            # Create a copy to avoid modifying original
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = color
            aligned_pred_scene.add_geometry(mesh_copy, node_name=f"aligned_pred_obj_{i}")
            comparison_scene.add_geometry(mesh_copy, node_name=f"aligned_pred_obj_{i}")
        
        # Add ground truth meshes to scene with same colors but different transparency
        for i, mesh in enumerate(gt_meshes):
            if i < len(colors):
                # Same color but with 40% opacity (more transparent)
                color = colors[i].copy()
                color[3] = 0.4
            else:
                # Generate a color if we have more objects than predefined colors
                hue = (i * 137.508) % 360
                color = [hue/360, 0.8, 0.8, 0.4]  # HSV to RGB approximation with 40% opacity
            
            # Create a copy to avoid modifying original
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = color
            gt_scene.add_geometry(mesh_copy, node_name=f"gt_obj_{i}")
            comparison_scene.add_geometry(mesh_copy, node_name=f"gt_obj_{i}")
        
        # Save individual scenes
        aligned_pred_path = results_dir / "aligned_pred_visualization.glb"
        gt_path = results_dir / "gt_visualization.glb"
        comparison_path = results_dir / "comparison_visualization.glb"
        
        # Export as GLB files
        aligned_pred_scene.export(str(aligned_pred_path))
        gt_scene.export(str(gt_path))
        comparison_scene.export(str(comparison_path))
        
        print(f"    Saved aligned_pred_visualization.glb: {aligned_pred_path}")
        print(f"    Saved gt_visualization.glb: {gt_path}")
        print(f"    Saved comparison_visualization.glb: {comparison_path}")
        
        # Also save as PLY for compatibility
        aligned_pred_ply = results_dir / "aligned_pred_visualization.ply"
        gt_ply = results_dir / "gt_visualization.ply"
        comparison_ply = results_dir / "comparison_visualization.ply"
        
        aligned_pred_scene.export(str(aligned_pred_ply))
        gt_scene.export(str(gt_ply))
        comparison_scene.export(str(comparison_ply))
        
        print(f"    Saved PLY files: {aligned_pred_ply.name}, {gt_ply.name}, {comparison_ply.name}")
        
        return True
        
    except Exception as e:
        print(f"    Error creating real scene visualization: {e}")
        return False

def evaluate_real_scene_penetration(aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh, scene_dir):
    """
    Evaluate penetration metrics for real scene data
    
    Args:
        aligned_pred_meshes: List of aligned predicted meshes
        gt_meshes: List of ground truth meshes
        merged_aligned_pred_mesh: Merged aligned predicted mesh
        merged_gt_mesh: Merged ground truth mesh
        scene_dir: Scene directory path
        
    Returns:
        dict: Comprehensive evaluation metrics
    """
    print(f"  Evaluating real scene penetration for: {scene_dir.name}")
    
    try:
        # Initialize metrics dictionary
        metrics = {}
        
        # Compute object-level metrics using individual meshes
        print(f"    Computing object-level metrics from individual meshes...")
        try:
            object_metrics = compute_multi_object_completion_metrics_messy_kitchen(
                gt_meshes=gt_meshes,
                pred_meshes=aligned_pred_meshes
            )
            
            print(f"    Object-level metrics:")
            print(f"      O-CD: {object_metrics['chamfer_distance']:.6f} ± {object_metrics['chamfer_distance_std']:.6f}")
            print(f"      O-IoU: {object_metrics['iou']:.6f} ± {object_metrics['iou_std']:.6f}")
            if 'b_iou' in object_metrics:
                print(f"      O-BIoU: {object_metrics['b_iou']:.6f} ± {object_metrics['b_iou_std']:.6f}")
            if 'f_score' in object_metrics:
                print(f"      O-Fscore: {object_metrics['f_score']:.6f} ± {object_metrics['f_score_std']:.6f}")
            
            metrics['object_level_metrics'] = object_metrics
            
        except Exception as e:
            print(f"      Error computing object-level metrics: {e}")
            metrics['object_level_metrics'] = None
        
        # Compute scene-level metrics using merged meshes
        print(f"    Computing scene-level metrics from merged meshes...")
        try:
            scene_cd, scene_iou, scene_b_iou, scene_fscore = compute_chamfer_and_iou_messy_kitchen(
                merged_gt_mesh, merged_aligned_pred_mesh
            )
            
            scene_metrics = {
                'chamfer_distance': scene_cd,
                'iou': scene_iou,
                'b_iou': scene_b_iou,
                'f_score': scene_fscore
            }
            
            print(f"    Scene-level metrics:")
            print(f"      S-CD: {scene_cd:.6f}")
            print(f"      S-IoU: {scene_iou:.6f}")
            print(f"      S-BIoU: {scene_b_iou:.6f}")
            print(f"      S-Fscore: {scene_fscore:.6f}")
            
            metrics['scene_level_metrics'] = scene_metrics
            
        except Exception as e:
            print(f"      Error computing scene-level metrics: {e}")
            metrics['scene_level_metrics'] = None
        
        # Calculate penetration levels with high resolution
        print(f"    Computing high-resolution penetration analysis...")
        penetration_metrics = calculate_real_scene_penetration_levels(
            aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh, scene_dir
        )
        
        if penetration_metrics:
            metrics['penetration_analysis'] = penetration_metrics
        
        return metrics
        
    except Exception as e:
        print(f"    Error evaluating real scene: {e}")
        return {}

def calculate_real_scene_penetration_levels(aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh, scene_dir):
    """
    Calculate penetration levels for real scene using high-resolution TSDF
    
    Args:
        aligned_pred_meshes: List of aligned predicted meshes
        gt_meshes: List of ground truth meshes
        merged_aligned_pred_mesh: Merged aligned predicted mesh
        merged_gt_mesh: Merged ground truth mesh
        scene_dir: Scene directory path
        
    Returns:
        dict: Penetration level metrics
    """
    try:
        print(f"      Calculating high-resolution penetration levels...")
        print(f"      Using TSDF resolution: 80x80x80 = 512K voxels")
        
        # Initialize penetration metrics
        penetration_metrics = {}
        
        # For aligned predicted meshes
        if aligned_pred_meshes and merged_aligned_pred_mesh is not None:
            print(f"        Computing penetration for aligned predicted meshes...")
            try:
                pred_penetration = compute_penetration_level_detailed(
                    aligned_pred_meshes, merged_aligned_pred_mesh, size=0.3, resolution=80
                )
                penetration_metrics['aligned_predicted'] = pred_penetration
                
                print(f"          Aligned predicted penetration level: {pred_penetration['penetration_level']:.6f}")
                print(f"          Aligned predicted overlap ratio: {pred_penetration['overlap_ratio']:.6f}")
                print(f"          Aligned predicted merged internal points: {pred_penetration['merged_internal_points']}")
                print(f"          Aligned predicted individual internal points sum: {pred_penetration['individual_internal_points_sum']}")
                
            except Exception as e:
                print(f"          Error computing aligned predicted penetration: {e}")
                penetration_metrics['aligned_predicted'] = None
        
        # For ground truth meshes
        if gt_meshes and merged_gt_mesh is not None:
            print(f"        Computing penetration for GT meshes...")
            try:
                gt_penetration = compute_penetration_level_detailed(
                    gt_meshes, merged_gt_mesh, size=0.3, resolution=80
                )
                penetration_metrics['ground_truth'] = gt_penetration
                
                print(f"          GT penetration level: {gt_penetration['penetration_level']:.6f}")
                print(f"          GT overlap ratio: {gt_penetration['overlap_ratio']:.6f}")
                print(f"          GT merged internal points: {gt_penetration['merged_internal_points']}")
                print(f"          GT individual internal points sum: {gt_penetration['individual_internal_points_sum']}")
                
            except Exception as e:
                print(f"          Error computing GT penetration: {e}")
                penetration_metrics['ground_truth'] = None
        
        # Calculate B-IoU between aligned predicted and GT merged meshes
        if merged_aligned_pred_mesh is not None and merged_gt_mesh is not None:
            print(f"        Computing B-IoU between aligned predicted and GT merged meshes...")
            try:
                b_iou = compute_bounding_box_iou(merged_gt_mesh, merged_aligned_pred_mesh)
                penetration_metrics['b_iou'] = float(b_iou)
                print(f"          B-IoU (aligned pred vs GT): {b_iou:.6f}")
                
            except Exception as e:
                print(f"          Error computing B-IoU: {e}")
                penetration_metrics['b_iou'] = None
        
        # Store merged meshes for reference
        penetration_metrics['merged_aligned_pred'] = merged_aligned_pred_mesh
        penetration_metrics['merged_gt'] = merged_gt_mesh
        
        # Save penetration analysis results
        save_real_scene_penetration_results(penetration_metrics, scene_dir)
        
        return penetration_metrics
        
    except Exception as e:
        print(f"        Error calculating real scene penetration levels: {e}")
        return {}

def save_real_scene_penetration_results(penetration_metrics, scene_dir):
    """
    Save real scene penetration analysis results to files
    
    Args:
        penetration_metrics: Dictionary containing penetration metrics
        scene_dir: Scene directory path
    """
    try:
        # Create real_scene_results directory
        results_dir = scene_dir / "real_scene_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_file = results_dir / "real_scene_penetration_analysis.txt"
        with open(detailed_file, 'w') as f:
            f.write(f"Real Scene Penetration Analysis: {scene_dir.name}\n")
            f.write("="*60 + "\n\n")
            f.write("High-resolution TSDF analysis (80x80x80 = 512K voxels)\n\n")
            
            # Write B-IoU if available
            if 'b_iou' in penetration_metrics and penetration_metrics['b_iou'] is not None:
                f.write(f"BOUNDING BOX IoU (Aligned Pred vs GT):\n")
                f.write(f"  B-IoU: {penetration_metrics['b_iou']:.6f}\n\n")
            
            for mesh_type, metrics in penetration_metrics.items():
                if mesh_type in ['b_iou', 'merged_aligned_pred', 'merged_gt']:
                    continue  # Skip non-metric fields
                    
                if metrics is None:
                    f.write(f"{mesh_type.upper()} MESHES: Error in computation\n\n")
                    continue
                    
                f.write(f"{mesh_type.upper()} MESHES:\n")
                f.write(f"  Penetration Level: {metrics['penetration_level']:.6f}\n")
                f.write(f"  Overlap Ratio: {metrics['overlap_ratio']:.6f}\n")
                f.write(f"  Merged Internal Points: {metrics['merged_internal_points']}\n")
                f.write(f"  Individual Internal Points Sum: {metrics['individual_internal_points_sum']}\n")
                
                if 'per_mesh_internal_points' in metrics:
                    f.write(f"  Per-mesh Internal Points: {metrics['per_mesh_internal_points']}\n")
                if 'per_mesh_penetration' in metrics:
                    f.write(f"  Per-mesh Penetration: {[f'{p:.6f}' for p in metrics['per_mesh_penetration']]}\n")
                
                f.write("\n")
        
        # Save summary JSON
        summary_file = results_dir / "real_scene_penetration_summary.json"
        
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for mesh_type, metrics in penetration_metrics.items():
            if mesh_type in ['merged_aligned_pred', 'merged_gt']:
                # Skip mesh objects in JSON
                continue
            if metrics is None:
                json_metrics[mesh_type] = None
                continue
                
            json_metrics[mesh_type] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[mesh_type][key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_metrics[mesh_type][key] = int(value)
                elif isinstance(value, np.floating):
                    json_metrics[mesh_type][key] = float(value)
                else:
                    json_metrics[mesh_type][key] = value
        
        with open(summary_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"        Real scene penetration results saved to: {results_dir}")
        
    except Exception as e:
        print(f"        Error saving real scene penetration results: {e}")

def evaluate_real_scene_directory(base_dir):
    """
    Evaluate real scene data from the specified directory
    
    Args:
        base_dir (str): Base directory containing real scene data
    """
    base_path = Path(base_dir)
    real_scene_path = base_path / "data" / "messy_kitchen_real" / "demo-sc-1_aligned_pred"
    
    if not real_scene_path.exists():
        print(f"Error: Real scene directory not found at {real_scene_path}")
        return
    
    print(f"Processing real scene data from: {real_scene_path}")
    
    # Initialize lists to store metrics
    all_scene_metrics = []
    scene_names = []
    
    # Check if this is a single scene directory (contains mesh files directly)
    # or a directory containing multiple scene subdirectories
    mesh_files = list(real_scene_path.glob("*.ply")) + list(real_scene_path.glob("*.glb"))
    
    if mesh_files:
        # This is a single scene directory with mesh files
        print(f"Found single scene directory with {len(mesh_files)} mesh files")
        scene_dir = real_scene_path
        scene_name = scene_dir.name
        
        print(f"\nProcessing real scene: {scene_name}")
        
        # Load meshes for this scene
        aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh = load_real_scene_meshes(scene_dir)
        
        if not aligned_pred_meshes or not gt_meshes:
            print(f"  Skipping scene {scene_name}: No valid meshes found")
            return
        
        if UTILS_GIGA_AVAILABLE:
            try:
                # Evaluate penetration metrics
                metrics = evaluate_real_scene_penetration(
                    aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh, scene_dir
                )
                
                if metrics:
                    all_scene_metrics.append(metrics)
                    scene_names.append(scene_name)
                    
                    # Create visualization
                    create_real_scene_visualization(aligned_pred_meshes, gt_meshes, scene_dir, scene_name)
                    
                else:
                    print(f"  Warning: Could not compute metrics for scene {scene_name}")
                    return
                
            except Exception as e:
                print(f"  Error evaluating scene {scene_name}: {e}")
                return
        else:
            print(f"  Skipping penetration evaluation (utils_giga not available)")
            print(f"  Only mesh normalization and saving completed for scene {scene_name}")
            # Still create visualization even without utils_giga
            try:
                create_real_scene_visualization(aligned_pred_meshes, gt_meshes, scene_dir, scene_name)
            except Exception as e:
                print(f"  Warning: Could not create visualization: {e}")
    else:
        # This is a directory containing multiple scene subdirectories
        scene_dirs = [d for d in real_scene_path.iterdir() if d.is_dir()]
        print(f"Found {len(scene_dirs)} scene directories")
        
        for scene_dir in scene_dirs:
            print(f"\nProcessing real scene: {scene_dir.name}")
            
            # Load meshes for this scene
            aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh = load_real_scene_meshes(scene_dir)
            
            if not aligned_pred_meshes or not gt_meshes:
                print(f"  Skipping scene {scene_dir.name}: No valid meshes found")
                continue
            
            try:
                # Evaluate penetration metrics
                metrics = evaluate_real_scene_penetration(
                    aligned_pred_meshes, gt_meshes, merged_aligned_pred_mesh, merged_gt_mesh, scene_dir
                )
                
                if metrics:
                    all_scene_metrics.append(metrics)
                    scene_names.append(scene_dir.name)
                    
                    # Create visualization
                    create_real_scene_visualization(aligned_pred_meshes, gt_meshes, scene_dir, scene_dir.name)
                    
                else:
                    print(f"  Warning: Could not compute metrics for scene {scene_dir.name}")
                    continue
                
            except Exception as e:
                print(f"  Error evaluating scene {scene_dir.name}: {e}")
                continue
    
    # Calculate overall statistics
    if all_scene_metrics:
        calculate_real_scene_overall_statistics(all_scene_metrics, scene_names, real_scene_path)
    else:
        print("No valid scene metrics found")

def calculate_real_scene_overall_statistics(all_scene_metrics, scene_names, output_dir):
    """
    Calculate and display overall statistics from all real scene metrics
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"REAL SCENE OVERALL STATISTICS")
    print(f"{'='*60}")
    
    # Extract Object-level metrics
    o_cds = []
    o_ious = []
    o_b_ious = []
    o_fscores = []
    
    # Extract Scene-level metrics
    s_cds = []
    s_ious = []
    s_b_ious = []
    s_fscores = []
    
    # Extract Penetration metrics
    aligned_pred_penetrations = []
    gt_penetrations = []
    b_ious = []
    
    for m in all_scene_metrics:
        # Object-level metrics
        if 'object_level_metrics' in m and m['object_level_metrics']:
            obj_metrics = m['object_level_metrics']
            o_cds.append(obj_metrics['chamfer_distance'])
            o_ious.append(obj_metrics['iou'])
            if 'b_iou' in obj_metrics:
                o_b_ious.append(obj_metrics['b_iou'])
            if 'f_score' in obj_metrics:
                o_fscores.append(obj_metrics['f_score'])
        
        # Scene-level metrics
        if 'scene_level_metrics' in m and m['scene_level_metrics']:
            scene_metrics = m['scene_level_metrics']
            s_cds.append(scene_metrics['chamfer_distance'])
            s_ious.append(scene_metrics['iou'])
            if 'b_iou' in scene_metrics:
                s_b_ious.append(scene_metrics['b_iou'])
            if 'f_score' in scene_metrics:
                s_fscores.append(scene_metrics['f_score'])
        
        # Penetration metrics
        if 'penetration_analysis' in m:
            if 'b_iou' in m['penetration_analysis'] and m['penetration_analysis']['b_iou'] is not None:
                b_ious.append(m['penetration_analysis']['b_iou'])
            if 'aligned_predicted' in m['penetration_analysis'] and m['penetration_analysis']['aligned_predicted']:
                aligned_pred_penetrations.append(m['penetration_analysis']['aligned_predicted']['penetration_level'])
            if 'ground_truth' in m['penetration_analysis'] and m['penetration_analysis']['ground_truth']:
                gt_penetrations.append(m['penetration_analysis']['ground_truth']['penetration_level'])
    
    # Display statistics
    print(f"\n{'='*60}")
    print(f"OBJECT-LEVEL METRICS (O-*)")
    print(f"{'='*60}")
    
    if o_cds:
        print(f"O-CD (Object-level Chamfer Distance): {np.mean(o_cds):.6f} ± {np.std(o_cds):.6f}")
        print(f"  Range: [{np.min(o_cds):.6f}, {np.max(o_cds):.6f}], Valid scenes: {len(o_cds)}")
    
    if o_ious:
        print(f"O-IoU (Object-level IoU): {np.mean(o_ious):.6f} ± {np.std(o_ious):.6f}")
        print(f"  Range: [{np.min(o_ious):.6f}, {np.max(o_ious):.6f}], Valid scenes: {len(o_ious)}")
    
    if o_b_ious:
        print(f"O-BIoU (Object-level Bounding-box IoU): {np.mean(o_b_ious):.6f} ± {np.std(o_b_ious):.6f}")
        print(f"  Range: [{np.min(o_b_ious):.6f}, {np.max(o_b_ious):.6f}], Valid scenes: {len(o_b_ious)}")
    
    if o_fscores:
        print(f"O-Fscore (Object-level F-score): {np.mean(o_fscores):.6f} ± {np.std(o_fscores):.6f}")
        print(f"  Range: [{np.min(o_fscores):.6f}, {np.max(o_fscores):.6f}], Valid scenes: {len(o_fscores)}")
    
    print(f"\n{'='*60}")
    print(f"SCENE-LEVEL METRICS (S-*)")
    print(f"{'='*60}")
    
    if s_cds:
        print(f"S-CD (Scene-level Chamfer Distance): {np.mean(s_cds):.6f} ± {np.std(s_cds):.6f}")
        print(f"  Range: [{np.min(s_cds):.6f}, {np.max(s_cds):.6f}], Valid scenes: {len(s_cds)}")
    
    if s_ious:
        print(f"S-IoU (Scene-level IoU): {np.mean(s_ious):.6f} ± {np.std(s_ious):.6f}")
        print(f"  Range: [{np.min(s_ious):.6f}, {np.max(s_ious):.6f}], Valid scenes: {len(s_ious)}")
    
    if s_b_ious:
        print(f"S-BIoU (Scene-level Bounding-box IoU): {np.mean(s_b_ious):.6f} ± {np.std(s_b_ious):.6f}")
        print(f"  Range: [{np.min(s_b_ious):.6f}, {np.max(s_b_ious):.6f}], Valid scenes: {len(s_b_ious)}")
    
    if s_fscores:
        print(f"S-Fscore (Scene-level F-score): {np.mean(s_fscores):.6f} ± {np.std(s_fscores):.6f}")
        print(f"  Range: [{np.min(s_fscores):.6f}, {np.max(s_fscores):.6f}], Valid scenes: {len(s_fscores)}")
    
    print(f"\n{'='*60}")
    print(f"PENETRATION ANALYSIS (High-Resolution TSDF)")
    print(f"{'='*60}")
    
    if b_ious:
        print(f"B-IoU (Aligned Pred vs GT): {np.mean(b_ious):.6f} ± {np.std(b_ious):.6f}")
        print(f"  Range: [{np.min(b_ious):.6f}, {np.max(b_ious):.6f}], Valid scenes: {len(b_ious)}")
    
    if aligned_pred_penetrations:
        print(f"Aligned Predicted Penetration: {np.mean(aligned_pred_penetrations):.6f} ± {np.std(aligned_pred_penetrations):.6f}")
        print(f"  Range: [{np.min(aligned_pred_penetrations):.6f}, {np.max(aligned_pred_penetrations):.6f}], Valid scenes: {len(aligned_pred_penetrations)}")
    
    if gt_penetrations:
        print(f"Ground Truth Penetration: {np.mean(gt_penetrations):.6f} ± {np.std(gt_penetrations):.6f}")
        print(f"  Range: [{np.min(gt_penetrations):.6f}, {np.max(gt_penetrations):.6f}], Valid scenes: {len(gt_penetrations)}")
    
    # Save overall results
    save_real_scene_overall_results(all_scene_metrics, scene_names, output_dir)

def save_real_scene_overall_results(all_scene_metrics, scene_names, output_dir):
    """
    Save overall real scene results to files
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    try:
        # Create overall results directory
        overall_dir = output_dir / "overall_real_scene_results"
        overall_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_file = overall_dir / "real_scene_detailed_metrics.txt"
        with open(detailed_file, 'w') as f:
            f.write("Real Scene Detailed Metrics\n")
            f.write("="*50 + "\n\n")
            f.write("High-resolution TSDF analysis (80x80x80 = 512K voxels)\n\n")
            
            for i, (metrics, scene_name) in enumerate(zip(all_scene_metrics, scene_names)):
                f.write(f"Scene {i+1}: {scene_name}\n")
                
                # Object-level metrics
                if 'object_level_metrics' in metrics and metrics['object_level_metrics']:
                    obj_metrics = metrics['object_level_metrics']
                    f.write(f"  O-CD: {obj_metrics['chamfer_distance']:.6f} ± {obj_metrics['chamfer_distance_std']:.6f}\n")
                    f.write(f"  O-IoU: {obj_metrics['iou']:.6f} ± {obj_metrics['iou_std']:.6f}\n")
                    if 'b_iou' in obj_metrics:
                        f.write(f"  O-BIoU: {obj_metrics['b_iou']:.6f} ± {obj_metrics['b_iou_std']:.6f}\n")
                    if 'f_score' in obj_metrics:
                        f.write(f"  O-Fscore: {obj_metrics['f_score']:.6f} ± {obj_metrics['f_score_std']:.6f}\n")
                
                # Scene-level metrics
                if 'scene_level_metrics' in metrics and metrics['scene_level_metrics']:
                    scene_metrics = metrics['scene_level_metrics']
                    f.write(f"  S-CD: {scene_metrics['chamfer_distance']:.6f}\n")
                    f.write(f"  S-IoU: {scene_metrics['iou']:.6f}\n")
                    if 'b_iou' in scene_metrics:
                        f.write(f"  S-BIoU: {scene_metrics['b_iou']:.6f}\n")
                    if 'f_score' in scene_metrics:
                        f.write(f"  S-Fscore: {scene_metrics['f_score']:.6f}\n")
                
                # Penetration metrics
                if 'penetration_analysis' in metrics:
                    if 'b_iou' in metrics['penetration_analysis'] and metrics['penetration_analysis']['b_iou'] is not None:
                        f.write(f"  B-IoU: {metrics['penetration_analysis']['b_iou']:.6f}\n")
                    if 'aligned_predicted' in metrics['penetration_analysis'] and metrics['penetration_analysis']['aligned_predicted']:
                        pred_pen = metrics['penetration_analysis']['aligned_predicted']
                        f.write(f"  Aligned Pred Penetration: {pred_pen['penetration_level']:.6f}\n")
                    if 'ground_truth' in metrics['penetration_analysis'] and metrics['penetration_analysis']['ground_truth']:
                        gt_pen = metrics['penetration_analysis']['ground_truth']
                        f.write(f"  GT Penetration: {gt_pen['penetration_level']:.6f}\n")
                
                f.write("\n")
        
        print(f"\nOverall real scene results saved to: {overall_dir}")
        
    except Exception as e:
        print(f"Error saving overall real scene results: {e}")

def main():
    """
    Main function to run real scene penetration evaluation
    """
    print("="*60)
    print("REAL SCENE PENETRATION SCORE EVALUATION")
    print("="*60)
    print("Processing real scene data from demo-sc-1_aligned_pred")
    print("High-resolution TSDF analysis (80x80x80 = 512K voxels)")
    print("="*60)
    
    # Path to real scene data
    base_dir = "/home/ran.ding/projects/TARGO"
    
    try:
        # Evaluate real scene data
        evaluate_real_scene_directory(base_dir)
        
        print(f"\n{'='*60}")
        print("REAL SCENE EVALUATION COMPLETED")
        print(f"{'='*60}")
        print("✓ High-resolution penetration analysis completed")
        print("✓ Object-level and scene-level metrics computed")
        print("✓ Visualizations and results saved")
        print("✓ Check individual scene directories for detailed results")
        
    except Exception as e:
        print(f"\n✗ Real scene evaluation failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
