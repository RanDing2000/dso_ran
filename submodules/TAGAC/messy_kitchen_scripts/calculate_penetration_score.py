import os
import json
from pathlib import Path
import numpy as np
import trimesh
import sys

# Add the src directory to the path to import utils_giga
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils_giga import (
    compute_multi_object_completion_metrics_messy_kitchen,
    compute_chamfer_and_iou_messy_kitchen,
    compute_penetration_level,
    compute_penetration_level_detailed,
    mesh_to_tsdf_messy_kitchen,
    compute_bounding_box_iou
)

def create_matched_visualization(pred_meshes, gt_meshes, scene_dir, scene_name):
    """
    Create visualization of matched meshes with same colors but different transparency
    
    Args:
        pred_meshes: List of predicted meshes
        gt_meshes: List of ground truth meshes  
        scene_dir: Scene directory path
        scene_name: Name of the scene
    """
    try:
        # Create matching_results directory
        matching_dir = scene_dir / "matching_results"
        matching_dir.mkdir(exist_ok=True)
        
        print(f"  Creating matched visualization in: {matching_dir}")
        
        # Create matched scenes with same colors but different transparency
        matched_pred_scene = trimesh.Scene()
        matched_gt_scene = trimesh.Scene()
        
        # Color palette for different objects (same color, different transparency)
        colors = [
            [1.0, 0.0, 0.0, 0.8],  # Red with 80% opacity
            [0.0, 1.0, 0.0, 0.8],  # Green with 80% opacity  
            [0.0, 0.0, 1.0, 0.8],  # Blue with 80% opacity
            [1.0, 1.0, 0.0, 0.8],  # Yellow with 80% opacity
            [1.0, 0.0, 1.0, 0.8],  # Magenta with 80% opacity
            [0.0, 1.0, 1.0, 0.8],  # Cyan with 80% opacity
        ]
        
        # Add predicted meshes to scene
        for i, mesh in enumerate(pred_meshes):
            if i < len(colors):
                color = colors[i]
            else:
                # Generate a color if we have more objects than predefined colors
                hue = (i * 137.508) % 360  # Golden angle approximation
                color = [hue/360, 0.8, 0.8, 0.8]  # HSV to RGB approximation
            
            # Create a copy to avoid modifying original
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = color
            matched_pred_scene.add_geometry(mesh_copy, node_name=f"pred_obj_{i}")
        
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
            matched_gt_scene.add_geometry(mesh_copy, node_name=f"gt_obj_{i}")
        
        # Save matched scenes
        matched_pred_path = matching_dir / "matched_pred_merged.glb"
        matched_gt_path = matching_dir / "matched_gt_merged.glb"
        
        # Export as GLB files
        matched_pred_scene.export(str(matched_pred_path))
        matched_gt_scene.export(str(matched_gt_path))
        
        print(f"    Saved matched_pred_merged.glb: {matched_pred_path}")
        print(f"    Saved matched_gt_merged.glb: {matched_gt_path}")
        
        # Also save as PLY for compatibility
        matched_pred_ply = matching_dir / "matched_pred_merged.ply"
        matched_gt_ply = matching_dir / "matched_gt_merged.ply"
        
        matched_pred_scene.export(str(matched_pred_ply))
        matched_gt_scene.export(str(matched_gt_ply))
        
        print(f"    Saved matched_pred_merged.ply: {matched_pred_ply}")
        print(f"    Saved matched_gt_merged.ply: {matched_gt_ply}")
        
        return True
        
    except Exception as e:
        print(f"    Error creating matched visualization: {e}")
        return False

def load_meshes_from_directory(scene_dir):
    """
    Load both individual and merged predicted and ground truth meshes from a scene directory
    
    Args:
        scene_dir (Path): Path to scene directory
        
    Returns:
        tuple: (pred_meshes, gt_meshes, merged_pred_mesh, merged_gt_mesh)
    """
    pred_meshes = []
    gt_meshes = []
    merged_pred_mesh = None
    merged_gt_mesh = None
    
    try:
        # Load individual predicted meshes (aligned_pred_obj_*.ply)
        pred_files = sorted(scene_dir.glob("aligned_pred_obj_*.ply"))
        for pred_file in pred_files:
            try:
                mesh = trimesh.load_mesh(str(pred_file))
                # if not mesh.is_watertight:
                    # print(f"  Warning: Predicted mesh {pred_file.name} is not watertight")
                    # mesh.fill_holes()
                    # mesh.remove_degenerate_faces()
                    # mesh.remove_duplicate_faces()
                    # mesh.remove_infinite_values()
                    # mesh.remove_unreferenced_vertices()
                pred_meshes.append(mesh)
                print(f"  Loaded predicted mesh: {pred_file.name}")
            except Exception as e:
                print(f"  Error loading {pred_file.name}: {e}")
        
        # Load individual ground truth meshes (gt_object_*.ply)
        gt_files = sorted(scene_dir.glob("gt_object_*.ply"))
        for gt_file in gt_files:
            try:
                mesh = trimesh.load_mesh(str(gt_file))
                # if not mesh.is_watertight:
                    # print(f"  Warning: GT mesh {gt_file.name} is not watertight")
                    # mesh.fill_holes()
                    # mesh.remove_degenerate_faces()
                    # mesh.remove_duplicate_faces()
                    # mesh.remove_infinite_values()
                    # mesh.remove_unreferenced_vertices()
                gt_meshes.append(mesh)
                print(f"  Loaded GT mesh: {gt_file.name}")
            except Exception as e:
                print(f"  Error loading {gt_file.name}: {e}")
        
        # Load merged predicted mesh (aligned_pred_merged.glb)
        pred_merged_file = scene_dir / "aligned_pred_merged.glb"
        if pred_merged_file.exists():
            try:
                mesh = trimesh.load_mesh(str(pred_merged_file))
                # if not mesh.is_watertight:
                #     print(f"  Warning: Merged predicted mesh {pred_merged_file.name} is not watertight")
                #     mesh.fill_holes()
                #     mesh.remove_degenerate_faces()
                #     mesh.remove_duplicate_faces()
                #     mesh.remove_infinite_values()
                #     mesh.remove_unreferenced_vertices()
                merged_pred_mesh = mesh
                print(f"  Loaded merged predicted mesh: {pred_merged_file.name}")
            except Exception as e:
                print(f"  Error loading {pred_merged_file.name}: {e}")
        else:
            print(f"  Warning: Merged predicted mesh file not found: {pred_merged_file}")
        
        # Load merged ground truth mesh (gt_merged.glb)
        gt_merged_file = scene_dir / "gt_merged.glb"
        if gt_merged_file.exists():
            try:
                # mesh = trimesh.load_mesh(str(gt_merged_file))
                # if not mesh.is_watertight:
                #     print(f"  Warning: Merged GT mesh {gt_merged_file.name} is not watertight")
                #     mesh.fill_holes()
                #     mesh.remove_degenerate_faces()
                #     mesh.remove_duplicate_faces()
                #     mesh.remove_infinite_values()
                #     mesh.remove_unreferenced_vertices()
                merged_gt_mesh = mesh
                print(f"  Loaded merged GT mesh: {gt_merged_file.name}")
            except Exception as e:
                print(f"  Error loading {gt_merged_file.name}: {e}")
        else:
            print(f"  Warning: Merged GT mesh file not found: {gt_merged_file}")
        
        print(f"  Scene {scene_dir.name}: {len(pred_meshes)} individual pred meshes, {len(gt_meshes)} individual GT meshes")
        print(f"    Merged pred mesh: {merged_pred_mesh is not None}, Merged GT mesh: {merged_gt_mesh is not None}")
        
    except Exception as e:
        print(f"  Error processing scene directory {scene_dir}: {e}")
    
    return pred_meshes, gt_meshes, merged_pred_mesh, merged_gt_mesh

def evaluate_messy_kitchen_scenes(base_dir):
    """
    Evaluate all messy kitchen scenes in the our_model directory
    
    Args:
        base_dir (str): Base directory containing our_model folder
    """
    base_path = Path(base_dir)
    our_model_path = base_path / "evaluation_objects_demo_test_100"
    
    if not our_model_path.exists():
        print(f"Error: our_model directory not found at {our_model_path}")
        return
    
    print(f"Processing messy kitchen results from: {our_model_path}")
    
    # Initialize lists to store metrics
    all_scene_metrics = []
    scene_names = []
    
    # Get all subdirectories in our_model
    scene_dirs = [d for d in our_model_path.iterdir() if d.is_dir()]
    print(f"Found {len(scene_dirs)} scene directories")
    
    for scene_dir in scene_dirs:
        print(f"\nProcessing scene: {scene_dir.name}")
        
        # Load meshes for this scene (both individual and merged)
        pred_meshes, gt_meshes, merged_pred_mesh, merged_gt_mesh = load_meshes_from_directory(scene_dir)
        
        if not pred_meshes or not gt_meshes:
            print(f"  Skipping scene {scene_dir.name}: No valid individual meshes found")
            continue
        
        if not merged_pred_mesh or not merged_gt_mesh:
            print(f"  Skipping scene {scene_dir.name}: No valid merged meshes found")
            continue
        
        try:
            # Compute object-level metrics using individual meshes
            print(f"  Computing object-level metrics from individual meshes...")
            try:
                object_metrics = compute_multi_object_completion_metrics_messy_kitchen(
                    gt_meshes=gt_meshes,
                    pred_meshes=pred_meshes
                )
                
                print(f"  Scene {scene_dir.name} object-level metrics:")
                print(f"    O-CD: {object_metrics['chamfer_distance']:.6f} ± {object_metrics['chamfer_distance_std']:.6f}")
                print(f"    O-IoU: {object_metrics['iou']:.6f} ± {object_metrics['iou_std']:.6f}")
                if 'b_iou' in object_metrics:
                    print(f"    O-BIoU: {object_metrics['b_iou']:.6f} ± {object_metrics['b_iou_std']:.6f}")
                if 'f_score' in object_metrics:
                    print(f"    O-Fscore: {object_metrics['f_score']:.6f} ± {object_metrics['f_score_std']:.6f}")
                
            except Exception as e:
                print(f"    Error computing object-level metrics: {e}")
                object_metrics = None
            merged_gt_mesh = trimesh.util.concatenate(gt_meshes)
            # merged_pred_mesh = trimesh.util.concatenate(pred_meshes)
        
            # Compute scene-level metrics using merged meshes
            print(f"  Computing scene-level metrics from merged meshes...")
            try:
                scene_cd, scene_iou, scene_b_iou, scene_fscore = compute_chamfer_and_iou_messy_kitchen(
                    merged_gt_mesh, merged_pred_mesh
                )
                
                scene_metrics = {
                    'chamfer_distance': scene_cd,
                    'iou': scene_iou,
                    'b_iou': scene_b_iou,
                    'f_score': scene_fscore
                }
                
                print(f"  Scene {scene_dir.name} scene-level metrics:")
                print(f"    S-CD: {scene_cd:.6f}")
                print(f"    S-IoU: {scene_iou:.6f}")
                print(f"    S-BIoU: {scene_b_iou:.6f}")
                print(f"    S-Fscore: {scene_fscore:.6f}")
                
            except Exception as e:
                print(f"    Error computing scene-level metrics: {e}")
                scene_metrics = None
            
            # Combine both metrics
            if object_metrics and scene_metrics:
                metrics = {
                    # Object-level metrics
                    'chamfer_distance': object_metrics['chamfer_distance'],
                    'iou': object_metrics['iou'],
                    'chamfer_distance_std': object_metrics['chamfer_distance_std'],
                    'iou_std': object_metrics['iou_std'],
                    'per_object_cds': object_metrics['per_object_cds'],
                    'per_object_ious': object_metrics['per_object_ious'],
                    
                    # Scene-level metrics
                    'scene_level_metrics': scene_metrics
                }
                
                # Add optional object-level metrics if available
                if 'b_iou' in object_metrics:
                    metrics['b_iou'] = object_metrics['b_iou']
                    metrics['b_iou_std'] = object_metrics['b_iou_std']
                    metrics['per_object_b_ious'] = object_metrics['per_object_b_ious']
                if 'f_score' in object_metrics:
                    metrics['f_score'] = object_metrics['f_score']
                    metrics['f_score_std'] = object_metrics['f_score_std']
                    metrics['per_object_fscores'] = object_metrics['per_object_fscores']
                
                # Store metrics
                all_scene_metrics.append(metrics)
                scene_names.append(scene_dir.name)
                
                # Create matched visualization
                create_matched_visualization(pred_meshes, gt_meshes, scene_dir, scene_dir.name)
                
                # Calculate penetration levels
                # penetration_metrics = calculate_scene_penetration_levels([merged_pred_mesh], [merged_gt_mesh], scene_dir)
                penetration_metrics = calculate_scene_penetration_levels(pred_meshes, gt_meshes, scene_dir)
                # ompute_penetration_level_detailed(mesh_list, merged_mesh, size=0.3, resolution=40):

                # Store penetration metrics in the main metrics
                if penetration_metrics:
                    metrics['penetration_analysis'] = penetration_metrics
                    
            else:
                print(f"  Warning: Could not compute both object-level and scene-level metrics")
                continue
            
        except Exception as e:
            print(f"  Error evaluating scene {scene_dir.name}: {e}")
            continue
    
    # Calculate overall statistics
    if all_scene_metrics:
        calculate_overall_statistics(all_scene_metrics, scene_names, our_model_path)
    else:
        print("No valid scene metrics found")

def calculate_scene_penetration_levels(pred_meshes, gt_meshes, scene_dir):
    """
    Calculate penetration levels for a scene using merged meshes
    
    Args:
        pred_meshes: List containing single merged predicted mesh
        gt_meshes: List containing single merged ground truth mesh
        scene_dir: Scene directory path
        
    Returns:
        dict: Penetration level metrics
    """
    try:
        print(f"  Calculating penetration levels for scene: {scene_dir.name}")
        
        # Since we now have merged meshes directly, use them
        # pred_merged = pred_meshes[0] if pred_meshes else None
        # gt_merged = gt_meshes[0] if gt_meshes else None
        pred_merged = trimesh.util.concatenate(pred_meshes)
        gt_merged = trimesh.util.concatenate(gt_meshes)
        
        print(f"    Using merged predicted mesh: {pred_merged is not None}")
        print(f"    Using merged GT mesh: {gt_merged is not None}")
        
        # Calculate penetration levels using messy kitchen specific functions
        penetration_metrics = {}
        
        # For predicted meshes (single merged mesh)
        if pred_merged is not None:
            print(f"    Computing penetration for predicted merged mesh...")
            pred_penetration = compute_penetration_level_detailed(pred_meshes, pred_merged, size=0.3, resolution=40)
            # For a single merged mesh, we can't compute penetration level since we need individual meshes
            # Instead, we'll store the mesh information for potential future use
            pred_penetration = {
                'penetration_level': pred_penetration['penetration_level'],  # No penetration for single mesh
                'overlap_ratio': pred_penetration['overlap_ratio'],      # Full overlap with itself
                'merged_internal_points': pred_penetration['merged_internal_points'],  # Will be computed if needed
                'individual_internal_points_sum': pred_penetration['individual_internal_points_sum'],
                'per_mesh_internal_points': pred_penetration['per_mesh_internal_points'],
                'per_mesh_penetration': pred_penetration['per_mesh_penetration']
            }
            penetration_metrics['predicted'] = pred_penetration
            
            print(f"      Predicted penetration level: {pred_penetration['penetration_level']:.6f}")
            print(f"      Predicted overlap ratio: {pred_penetration['overlap_ratio']:.6f}")
        
        # For ground truth meshes (single merged mesh)
        if gt_merged is not None:
            print(f"    Computing penetration for GT merged mesh...")
            gt_penetration = compute_penetration_level_detailed(gt_meshes, gt_merged, size=0.3, resolution=40)
            # For a single merged mesh, we can't compute penetration level since we need individual meshes
            # Instead, we'll store the mesh information for potential future use
            gt_penetration = {
                'penetration_level': gt_penetration['penetration_level'],  # No penetration for single mesh
                'overlap_ratio': gt_penetration['overlap_ratio'],      # Full overlap with itself
                'merged_internal_points': gt_penetration['merged_internal_points'],  # Will be computed if needed
                'individual_internal_points_sum': gt_penetration['individual_internal_points_sum'],
                'per_mesh_internal_points': gt_penetration['per_mesh_internal_points'],
                'per_mesh_penetration': gt_penetration['per_mesh_penetration']
            }
            penetration_metrics['ground_truth'] = gt_penetration
            
            print(f"      GT penetration level: {gt_penetration['penetration_level']:.6f}")
            print(f"      GT overlap ratio: {gt_penetration['overlap_ratio']:.6f}")
        
        # Calculate B-IoU between predicted and GT merged meshes
        if pred_merged is not None and gt_merged is not None:
            print(f"    Computing B-IoU between predicted and GT merged meshes...")
            b_iou = compute_bounding_box_iou(gt_merged, pred_merged)
            penetration_metrics['b_iou'] = float(b_iou)
            print(f"      B-IoU (pred vs GT): {b_iou:.6f}")
        
        # Store merged meshes for scene-level computation
        penetration_metrics['merged_gt'] = gt_merged
        penetration_metrics['merged_pred'] = pred_merged
        
        # Save penetration analysis results
        save_penetration_results(penetration_metrics, scene_dir)
        
        return penetration_metrics
        
    except Exception as e:
        print(f"    Error calculating penetration levels: {e}")
        return {}

def save_penetration_results(penetration_metrics, scene_dir):
    """
    Save penetration analysis results to files
    
    Args:
        penetration_metrics: Dictionary containing penetration metrics
        scene_dir: Scene directory path
    """
    try:
        # Create penetration_results directory
        penetration_dir = scene_dir / "penetration_results"
        penetration_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        detailed_file = penetration_dir / "penetration_analysis.txt"
        with open(detailed_file, 'w') as f:
            f.write(f"Penetration Analysis for Scene: {scene_dir.name}\n")
            f.write("="*50 + "\n\n")
            
            # Write B-IoU if available
            if 'b_iou' in penetration_metrics:
                f.write(f"BOUNDING BOX IoU (Pred vs GT):\n")
                f.write(f"  B-IoU: {penetration_metrics['b_iou']:.6f}\n\n")
            
            for mesh_type, metrics in penetration_metrics.items():
                if mesh_type == 'b_iou':
                    continue  # Skip B-IoU as it's already written above
                    
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
        summary_file = penetration_dir / "penetration_summary.json"
        
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for mesh_type, metrics in penetration_metrics.items():
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
        
        print(f"    Penetration results saved to: {penetration_dir}")
        
    except Exception as e:
        print(f"    Error saving penetration results: {e}")

def calculate_overall_statistics(all_scene_metrics, scene_names, output_dir):
    """
    Calculate and display overall statistics from all scene metrics
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*60}")
    
    # Extract Object-level metrics (per-object averages)
    o_cds = [m['chamfer_distance'] for m in all_scene_metrics if m['chamfer_distance'] != float('inf')]
    o_ious = [m['iou'] for m in all_scene_metrics if m['iou'] > 0]
    o_b_ious = [m['b_iou'] for m in all_scene_metrics if 'b_iou' in m and m['b_iou'] > 0]
    o_fscores = [m['f_score'] for m in all_scene_metrics if 'f_score' in m and m['f_score'] > 0]
    
    # Extract Scene-level metrics (from scene_level_metrics)
    s_cds = []  # Scene-level Chamfer Distance
    s_ious = []  # Scene-level IoU
    s_b_ious = []  # Scene-level B-IoU
    s_fscores = []  # Scene-level F-score
    pred_penetrations = []
    gt_penetrations = []
    
    # Collect merged meshes for scene-level computation
    merged_gt_meshes = []
    merged_pred_meshes = []
    
    for m in all_scene_metrics:
        # Extract scene-level metrics from scene_level_metrics
        if 'scene_level_metrics' in m and m['scene_level_metrics']:
            scene_metrics = m['scene_level_metrics']
            s_cds.append(scene_metrics['chamfer_distance'])
            s_ious.append(scene_metrics['iou'])
            if 'b_iou' in scene_metrics:
                s_b_ious.append(scene_metrics['b_iou'])
            if 'f_score' in scene_metrics:
                s_fscores.append(scene_metrics['f_score'])
        
        # Extract penetration metrics
        if 'penetration_analysis' in m:
            if 'b_iou' in m['penetration_analysis']:
                # Use scene-level B-IoU if available, otherwise fall back to penetration analysis
                if 'scene_level_metrics' not in m or not m['scene_level_metrics']:
                    s_b_ious.append(m['penetration_analysis']['b_iou'])
            if 'predicted' in m['penetration_analysis']:
                pred_penetrations.append(m['penetration_analysis']['predicted']['penetration_level'])
            if 'ground_truth' in m['penetration_analysis']:
                gt_penetrations.append(m['penetration_analysis']['ground_truth']['penetration_level'])
            if 'merged_gt' in m['penetration_analysis']:
                merged_gt_meshes.append(m['penetration_analysis']['merged_gt'])
            if 'merged_pred' in m['penetration_analysis']:
                merged_pred_meshes.append(m['penetration_analysis']['merged_pred'])
    
    print(f"\n{'='*60}")
    print(f"OBJECT-LEVEL METRICS (O-*)")
    print(f"{'='*60}")
    
    # Object-level Chamfer Distance (O-CD)
    if o_cds:
        avg_o_cd = np.mean(o_cds)
        std_o_cd = np.std(o_cds)
        min_o_cd = np.min(o_cds)
        max_o_cd = np.max(o_cds)
        print(f"O-CD (Object-level Chamfer Distance):")
        print(f"  Average: {avg_o_cd:.6f} ± {std_o_cd:.6f}")
        print(f"  Range: [{min_o_cd:.6f}, {max_o_cd:.6f}]")
        print(f"  Valid scenes: {len(o_cds)}/{len(all_scene_metrics)}")
    else:
        print("O-CD (Object-level Chamfer Distance): No valid values")
    
    # Object-level IoU (O-IoU)
    if o_ious:
        avg_o_iou = np.mean(o_ious)
        std_o_iou = np.std(o_ious)
        min_o_iou = np.min(o_ious)
        max_o_iou = np.max(o_ious)
        print(f"\nO-IoU (Object-level IoU):")
        print(f"  Average: {avg_o_iou:.6f} ± {std_o_iou:.6f}")
        print(f"  Range: [{min_o_iou:.6f}, {max_o_iou:.6f}]")
        print(f"  Valid scenes: {len(o_ious)}/{len(all_scene_metrics)}")
    else:
        print("\nO-IoU (Object-level IoU): No valid values")
    
    # Object-level B-IoU (O-BIoU)
    if o_b_ious:
        avg_o_b_iou = np.mean(o_b_ious)
        std_o_b_iou = np.std(o_b_ious)
        min_o_b_iou = np.min(o_b_ious)
        max_o_b_iou = np.max(o_b_ious)
        print(f"\nO-BIoU (Object-level Bounding-box IoU):")
        print(f"  Average: {avg_o_b_iou:.6f} ± {std_o_b_iou:.6f}")
        print(f"  Range: [{min_o_b_iou:.6f}, {max_o_b_iou:.6f}]")
        print(f"  Valid scenes: {len(o_b_ious)}/{len(all_scene_metrics)}")
    else:
        print("\nO-BIoU (Object-level Bounding-box IoU): No valid values")
    
    # Object-level F-score (O-Fscore)
    if o_fscores:
        avg_o_fscore = np.mean(o_fscores)
        std_o_fscore = np.std(o_fscores)
        min_o_fscore = np.min(o_fscores)
        max_o_fscore = np.max(o_fscores)
        print(f"\nO-Fscore (Object-level F-score):")
        print(f"  Average: {avg_o_fscore:.6f} ± {std_o_fscore:.6f}")
        print(f"  Range: [{min_o_fscore:.6f}, {max_o_fscore:.6f}]")
        print(f"  Valid scenes: {len(o_fscores)}/{len(all_scene_metrics)}")
    else:
        print("\nO-Fscore (Object-level F-score): No valid values")
    
    print(f"\n{'='*60}")
    print(f"SCENE-LEVEL METRICS (S-*)")
    print(f"{'='*60}")
    
    # Scene-level Chamfer Distance (S-CD)
    if s_cds:
        avg_s_cd = np.mean(s_cds)
        std_s_cd = np.std(s_cds)
        min_s_cd = np.min(s_cds)
        max_s_cd = np.max(s_cds)
        print(f"S-CD (Scene-level Chamfer Distance):")
        print(f"  Average: {avg_s_cd:.6f} ± {std_s_cd:.6f}")
        print(f"  Range: [{min_s_cd:.6f}, {max_s_cd:.6f}]")
        print(f"  Valid scenes: {len(s_cds)}/{len(all_scene_metrics)}")
    else:
        print("S-CD (Scene-level Chamfer Distance): No valid values")
    
    # Scene-level IoU (S-IoU)
    if s_ious:
        avg_s_iou = np.mean(s_ious)
        std_s_iou = np.std(s_ious)
        min_s_iou = np.min(s_ious)
        max_s_iou = np.max(s_ious)
        print(f"\nS-IoU (Scene-level IoU):")
        print(f"  Average: {avg_s_iou:.6f} ± {std_s_iou:.6f}")
        print(f"  Range: [{min_s_iou:.6f}, {max_s_iou:.6f}]")
        print(f"  Valid scenes: {len(s_ious)}/{len(all_scene_metrics)}")
    else:
        print("\nS-IoU (Scene-level IoU): No valid values")
    
    # Scene-level B-IoU (S-BIoU)
    if s_b_ious:
        avg_s_b_iou = np.mean(s_b_ious)
        std_s_b_iou = np.std(s_b_ious)
        min_s_b_iou = np.min(s_b_ious)
        max_s_b_iou = np.max(s_b_ious)
        print(f"\nS-BIoU (Scene-level Bounding-box IoU):")
        print(f"  Average: {avg_s_b_iou:.6f} ± {std_s_b_iou:.6f}")
        print(f"  Range: [{min_s_b_iou:.6f}, {max_s_b_iou:.6f}]")
        print(f"  Valid scenes: {len(s_b_ious)}/{len(all_scene_metrics)}")
    else:
        print("\nS-BIoU (Scene-level Bounding-box IoU): No valid values")
    
    # Scene-level F-score (S-Fscore)
    if s_fscores:
        avg_s_fscore = np.mean(s_fscores)
        std_s_fscore = np.std(s_fscores)
        min_s_fscore = np.min(s_fscores)
        max_s_fscore = np.max(s_fscores)
        print(f"\nS-Fscore (Scene-level F-score):")
        print(f"  Average: {avg_s_fscore:.6f} ± {std_s_fscore:.6f}")
        print(f"  Range: [{min_s_fscore:.6f}, {max_s_fscore:.6f}]")
        print(f"  Valid scenes: {len(s_fscores)}/{len(all_scene_metrics)}")
    else:
        print("\nS-Fscore (Scene-level F-score): No valid values")
    
    # Scene-level penetration metrics
    if pred_penetrations:
        avg_pred_pen = np.mean(pred_penetrations)
        std_pred_pen = np.std(pred_penetrations)
        min_pred_pen = np.min(pred_penetrations)
        max_pred_pen = np.max(pred_penetrations)
        print(f"\nS-Penetration (Scene-level Predicted Penetration):")
        print(f"  Average: {avg_pred_pen:.6f} ± {std_pred_pen:.6f}")
        print(f"  Range: [{min_pred_pen:.6f}, {max_pred_pen:.6f}]")
        print(f"  Valid scenes: {len(pred_penetrations)}/{len(all_scene_metrics)}")
    else:
        print("\nS-Penetration (Scene-level Predicted Penetration): No valid values")
    
    if gt_penetrations:
        avg_gt_pen = np.mean(gt_penetrations)
        std_gt_pen = np.std(gt_penetrations)
        min_gt_pen = np.min(gt_penetrations)
        max_gt_pen = np.max(gt_penetrations)
        print(f"\nS-GT-Penetration (Scene-level GT Penetration):")
        print(f"  Average: {avg_gt_pen:.6f} ± {std_gt_pen:.6f}")
        print(f"  Range: [{min_gt_pen:.6f}, {max_gt_pen:.6f}]")
        print(f"  Valid scenes: {len(gt_penetrations)}/{len(all_scene_metrics)}")
    else:
        print("\nS-GT-Penetration (Scene-level GT Penetration): No valid values")
    
    # Scene-level mesh information
    if merged_gt_meshes and merged_pred_meshes:
        print(f"\nScene-level Merged Meshes:")
        print(f"  Available merged GT meshes: {len(merged_gt_meshes)}")
        print(f"  Available merged predicted meshes: {len(merged_pred_meshes)}")
        print(f"  Note: Scene-level metrics computed directly from merged meshes")
    
    # Save detailed results
    save_detailed_results(all_scene_metrics, scene_names, output_dir)
    
    # Save summary
    save_summary_results(all_scene_metrics, output_dir)

def save_detailed_results(all_scene_metrics, scene_names, output_dir):
    """
    Save detailed results for each scene
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        scene_names (list): List of scene names
        output_dir (Path): Directory to save results
    """
    detailed_file = output_dir / "detailed_metrics.txt"
    
    with open(detailed_file, 'w') as f:
        f.write("Detailed Metrics for Each Scene\n")
        f.write("="*50 + "\n\n")
        
        for i, (metrics, scene_name) in enumerate(zip(all_scene_metrics, scene_names)):
            f.write(f"Scene {i+1}: {scene_name}\n")
            
            # Object-level metrics
            f.write(f"  O-CD (Object-level CD): {metrics['chamfer_distance']:.6f} ± {metrics['chamfer_distance_std']:.6f}\n")
            f.write(f"  O-IoU (Object-level IoU): {metrics['iou']:.6f} ± {metrics['iou_std']:.6f}\n")
            
            if 'b_iou' in metrics:
                f.write(f"  O-BIoU (Object-level B-IoU): {metrics['b_iou']:.6f} ± {metrics['b_iou_std']:.6f}\n")
            if 'f_score' in metrics:
                f.write(f"  O-Fscore (Object-level F-score): {metrics['f_score']:.6f} ± {metrics['f_score_std']:.6f}\n")
            
            # Per-object detailed metrics
            if 'per_object_cds' in metrics and metrics['per_object_cds']:
                f.write(f"  Per-object CDs: {[f'{cd:.6f}' for cd in metrics['per_object_cds']]}\n")
            if 'per_object_ious' in metrics and metrics['per_object_ious']:
                f.write(f"  Per-object IoUs: {[f'{iou:.6f}' for iou in metrics['per_object_ious']]}\n")
            if 'per_object_b_ious' in metrics and metrics['per_object_b_ious']:
                f.write(f"  Per-object B-IoUs: {[f'{b_iou:.6f}' for b_iou in metrics['per_object_b_ious']]}\n")
            if 'per_object_fscores' in metrics and metrics['per_object_fscores']:
                f.write(f"  Per-object F-scores: {[f'{fscore:.6f}' for fscore in metrics['per_object_fscores']]}\n")
            
            # Scene-level metrics (from scene_level_metrics)
            if 'scene_level_metrics' in metrics and metrics['scene_level_metrics']:
                f.write(f"  Scene-level Metrics:\n")
                scene_metrics = metrics['scene_level_metrics']
                f.write(f"    S-CD (Scene-level CD): {scene_metrics['chamfer_distance']:.6f}\n")
                f.write(f"    S-IoU (Scene-level IoU): {scene_metrics['iou']:.6f}\n")
                if 'b_iou' in scene_metrics:
                    f.write(f"    S-BIoU (Scene-level B-IoU): {scene_metrics['b_iou']:.6f}\n")
                if 'f_score' in scene_metrics:
                    f.write(f"    S-Fscore (Scene-level F-score): {scene_metrics['f_score']:.6f}\n")
            
            # Scene-level metrics (penetration analysis)
            if 'penetration_analysis' in metrics:
                f.write(f"  Scene-level Analysis:\n")
                if 'b_iou' in metrics['penetration_analysis']:
                    f.write(f"    S-BIoU (Scene-level B-IoU): {metrics['penetration_analysis']['b_iou']:.6f}\n")
                for mesh_type, pen_metrics in metrics['penetration_analysis'].items():
                    if mesh_type in ['b_iou', 'merged_gt', 'merged_pred']:
                        continue  # Skip non-metric fields
                    if isinstance(pen_metrics, dict) and 'penetration_level' in pen_metrics:
                        f.write(f"    S-Penetration ({mesh_type}): penetration={pen_metrics['penetration_level']:.6f}, overlap={pen_metrics['overlap_ratio']:.6f}\n")
                    else:
                        f.write(f"    {mesh_type}: {type(pen_metrics).__name__}\n")
            
            f.write("\n")
    
    print(f"\nDetailed results saved to: {detailed_file}")

def save_summary_results(all_scene_metrics, output_dir):
    """
    Save summary results
    
    Args:
        all_scene_metrics (list): List of metrics dictionaries
        output_dir (Path): Directory to save results
    """
    summary_file = output_dir / "metrics_summary.txt"
    
    # Calculate overall statistics
    o_cds = [m['chamfer_distance'] for m in all_scene_metrics if m['chamfer_distance'] != float('inf')]
    o_ious = [m['iou'] for m in all_scene_metrics if m['iou'] > 0]
    o_b_ious = [m['b_iou'] for m in all_scene_metrics if 'b_iou' in m and m['b_iou'] > 0]
    o_fscores = [m['f_score'] for m in all_scene_metrics if 'f_score' in m and m['f_score'] > 0]
    
    # Scene-level metrics from scene_level_metrics and penetration analysis
    s_cds = []
    s_ious = []
    s_b_ious = []
    s_fscores = []
    pred_penetrations = []
    gt_penetrations = []
    
    # Collect merged meshes for scene-level computation
    merged_gt_meshes = []
    merged_pred_meshes = []
    
    for m in all_scene_metrics:
        # Extract scene-level metrics from scene_level_metrics
        if 'scene_level_metrics' in m and m['scene_level_metrics']:
            scene_metrics = m['scene_level_metrics']
            s_cds.append(scene_metrics['chamfer_distance'])
            s_ious.append(scene_metrics['iou'])
            if 'b_iou' in scene_metrics:
                s_b_ious.append(scene_metrics['b_iou'])
            if 'f_score' in scene_metrics:
                s_fscores.append(scene_metrics['f_score'])
        
        # Extract penetration metrics
        if 'penetration_analysis' in m:
            if 'b_iou' in m['penetration_analysis']:
                # Use scene-level B-IoU if available, otherwise fall back to penetration analysis
                if 'scene_level_metrics' not in m or not m['scene_level_metrics']:
                    s_b_ious.append(m['penetration_analysis']['b_iou'])
            if 'predicted' in m['penetration_analysis']:
                pred_penetrations.append(m['penetration_analysis']['predicted']['penetration_level'])
            if 'ground_truth' in m['penetration_analysis']:
                gt_penetrations.append(m['penetration_analysis']['ground_truth']['penetration_level'])
            if 'merged_gt' in m['penetration_analysis']:
                merged_gt_meshes.append(m['penetration_analysis']['merged_gt'])
            if 'merged_pred' in m['penetration_analysis']:
                merged_pred_meshes.append(m['penetration_analysis']['merged_pred'])
    
    with open(summary_file, 'w') as f:
        f.write("Messy Kitchen Evaluation Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total scenes processed: {len(all_scene_metrics)}\n")
        f.write(f"Scenes with valid O-CD: {len(o_cds)}\n")
        f.write(f"Scenes with valid O-IoU: {len(o_ious)}\n")
        f.write(f"Scenes with valid O-BIoU: {len(o_b_ious)}\n")
        f.write(f"Scenes with valid O-Fscore: {len(o_fscores)}\n")
        f.write(f"Scenes with valid S-CD: {len(s_cds)}\n")
        f.write(f"Scenes with valid S-IoU: {len(s_ious)}\n")
        f.write(f"Scenes with valid S-BIoU: {len(s_b_ious)}\n")
        f.write(f"Scenes with valid S-Fscore: {len(s_fscores)}\n")
        f.write(f"Scenes with valid S-Penetration: {len(pred_penetrations)}\n")
        f.write(f"Scenes with merged meshes: {len(merged_gt_meshes)} GT, {len(merged_pred_meshes)} Pred\n\n")
        
        # Object-level metrics summary
        f.write(f"OBJECT-LEVEL METRICS (O-*)\n")
        f.write(f"{'='*30}\n")
        
        if o_cds:
            f.write(f"O-CD (Object-level Chamfer Distance):\n")
            f.write(f"  Average: {np.mean(o_cds):.6f} ± {np.std(o_cds):.6f}\n")
            f.write(f"  Min: {np.min(o_cds):.6f}\n")
            f.write(f"  Max: {np.max(o_cds):.6f}\n\n")
        
        if o_ious:
            f.write(f"O-IoU (Object-level IoU):\n")
            f.write(f"  Average: {np.mean(o_ious):.6f} ± {np.std(o_ious):.6f}\n")
            f.write(f"  Min: {np.min(o_ious):.6f}\n")
            f.write(f"  Max: {np.max(o_ious):.6f}\n\n")
        
        if o_b_ious:
            f.write(f"O-BIoU (Object-level Bounding-box IoU):\n")
            f.write(f"  Average: {np.mean(o_b_ious):.6f} ± {np.std(o_b_ious):.6f}\n")
            f.write(f"  Min: {np.min(o_b_ious):.6f}\n")
            f.write(f"  Max: {np.max(o_b_ious):.6f}\n\n")
        
        if o_fscores:
            f.write(f"O-Fscore (Object-level F-score):\n")
            f.write(f"  Average: {np.mean(o_fscores):.6f} ± {np.std(o_fscores):.6f}\n")
            f.write(f"  Min: {np.min(o_fscores):.6f}\n")
            f.write(f"  Max: {np.max(o_fscores):.6f}\n\n")
        
        # Scene-level metrics summary
        f.write(f"SCENE-LEVEL METRICS (S-*)\n")
        f.write(f"{'='*30}\n")
        
        if s_cds:
            f.write(f"S-CD (Scene-level Chamfer Distance):\n")
            f.write(f"  Average: {np.mean(s_cds):.6f} ± {np.std(s_cds):.6f}\n")
            f.write(f"  Min: {np.min(s_cds):.6f}\n")
            f.write(f"  Max: {np.max(s_cds):.6f}\n\n")
        
        if s_ious:
            f.write(f"S-IoU (Scene-level IoU):\n")
            f.write(f"  Average: {np.mean(s_ious):.6f} ± {np.std(s_ious):.6f}\n")
            f.write(f"  Min: {np.min(s_ious):.6f}\n")
            f.write(f"  Max: {np.max(s_ious):.6f}\n\n")
        
        if s_b_ious:
            f.write(f"S-BIoU (Scene-level Bounding-box IoU):\n")
            f.write(f"  Average: {np.mean(s_b_ious):.6f} ± {np.std(s_b_ious):.6f}\n")
            f.write(f"  Min: {np.min(s_b_ious):.6f}\n")
            f.write(f"  Max: {np.max(s_b_ious):.6f}\n\n")
        
        if s_fscores:
            f.write(f"S-Fscore (Scene-level F-score):\n")
            f.write(f"  Average: {np.mean(s_fscores):.6f} ± {np.std(s_fscores):.6f}\n")
            f.write(f"  Min: {np.min(s_fscores):.6f}\n")
            f.write(f"  Max: {np.max(s_fscores):.6f}\n\n")
        
        if pred_penetrations:
            f.write(f"S-Penetration (Scene-level Predicted Penetration):\n")
            f.write(f"  Average: {np.mean(pred_penetrations):.6f} ± {np.std(pred_penetrations):.6f}\n")
            f.write(f"  Min: {np.min(pred_penetrations):.6f}\n")
            f.write(f"  Max: {np.max(pred_penetrations):.6f}\n\n")
        
        if gt_penetrations:
            f.write(f"S-GT-Penetration (Scene-level GT Penetration):\n")
            f.write(f"  Average: {np.mean(gt_penetrations):.6f} ± {np.std(gt_penetrations):.6f}\n")
            f.write(f"  Min: {np.min(gt_penetrations):.6f}\n")
            f.write(f"  Max: {np.max(gt_penetrations):.6f}\n\n")
        
        # Scene-level mesh information
        if merged_gt_meshes and merged_pred_meshes:
            f.write(f"Scene-level Merged Meshes:\n")
            f.write(f"  Available merged GT meshes: {len(merged_gt_meshes)}\n")
            f.write(f"  Available merged predicted meshes: {len(merged_pred_meshes)}\n")
            f.write(f"  Note: Scene-level metrics computed directly from merged meshes\n")
    
    print(f"Summary results saved to: {summary_file}")

def calculate_average_metrics(base_dir):
    """
    Calculate average IoU and Chamfer L1 distance from all scene_metadata.txt files
    
    Args:
        base_dir (str): Base directory containing scene folders
    """
    # Initialize lists to store metrics
    iou_values = []
    cd_values = []
    
    # Get all subdirectories
    base_path = Path(base_dir)
    
    try:
        # Iterate through all subdirectories
        for scene_dir in base_path.iterdir():
            if not scene_dir.is_dir():
                continue
                
            metadata_file = scene_dir / "scene_metadata.txt"
            if not metadata_file.exists():
                continue
                
            # Read and parse metadata
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.loads(f.read())
                    
                # Extract metrics
                if 'iou' in metadata and 'cd' in metadata:
                    iou_values.append(metadata['iou'])
                    cd_values.append(metadata['cd'])
            except json.JSONDecodeError as e:
                print(f"Error parsing {metadata_file}: {e}")
            except Exception as e:
                print(f"Error processing {metadata_file}: {e}")
        
        # Calculate averages
        if iou_values and cd_values:
            avg_iou = np.mean(iou_values)
            avg_cd = np.mean(cd_values)
            
            # Calculate standard deviations
            std_iou = np.std(iou_values)
            std_cd = np.std(cd_values)
            
            print(f"\n总场景数: {len(iou_values)}")
            print(f"\n平均指标:")
            print(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
            print(f"Average Chamfer L1: {avg_cd:.4f} ± {std_cd:.4f}")
            
            # Save results to a summary file
            summary_path = os.path.join(base_dir, "metrics_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Total scenes: {len(iou_values)}\n")
                f.write(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\n")
                f.write(f"Average Chamfer L1: {avg_cd:.4f} ± {std_cd:.4f}\n")
            
            return avg_iou, avg_cd, len(iou_values)
        else:
            print("No valid metrics found in any scene_metadata.txt files")
            return None, None, 0
            
    except Exception as e:
        print(f"Error processing directory {base_dir}: {e}")
        return None, None, 0

if __name__ == "__main__":
    # Path to messy kitchen results
    base_dir = "/home/ran.ding/projects/TARGO/messy_kitchen_results"
    
    print("Messy Kitchen Scene Evaluation")
    print("="*50)
    
    # Evaluate messy kitchen scenes
    evaluate_messy_kitchen_scenes(base_dir)
    
    print(f"\nEvaluation completed for: {base_dir}") 
    print(f"\nEvaluation completed for: {base_dir}") 