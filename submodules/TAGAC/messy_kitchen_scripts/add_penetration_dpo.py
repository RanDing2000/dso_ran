import os
import json
from pathlib import Path
import numpy as np
import trimesh
import sys

# Add the src directory to the path to import utils_giga
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from utils_giga import (
    compute_penetration_level_detailed,
    mesh_to_tsdf_messy_kitchen
)

def load_dpo_meshes_from_seed_dir(seed_dir):
    """
    Load predicted meshes from a DPO seed directory
    
    Args:
        seed_dir (Path): Path to seed directory (e.g., 0/, 123/, etc.)
        
    Returns:
        tuple: (pred_meshes, merged_pred_mesh)
    """
    pred_meshes = []
    merged_pred_mesh = None
    
    try:
        # Load individual predicted meshes (pred_obj_*.ply)
        pred_files = sorted(seed_dir.glob("pred_obj_*.ply"))
        for pred_file in pred_files:
            try:
                mesh = trimesh.load_mesh(str(pred_file))
                pred_meshes.append(mesh)
                print(f"    Loaded predicted mesh: {pred_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_file.name}: {e}")
        
        # Load merged predicted mesh (pred_merged.glb)
        pred_merged_file = seed_dir / "pred_merged.glb"
        if pred_merged_file.exists():
            try:
                mesh = trimesh.load_mesh(str(pred_merged_file))
                merged_pred_mesh = mesh
                print(f"    Loaded merged predicted mesh: {pred_merged_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_merged_file.name}: {e}")
        else:
            print(f"    Warning: Merged predicted mesh file not found: {pred_merged_file}")
        
        print(f"    Seed {seed_dir.name}: {len(pred_meshes)} individual pred meshes")
        print(f"      Merged pred mesh: {merged_pred_mesh is not None}")
        
    except Exception as e:
        print(f"    Error processing seed directory {seed_dir}: {e}")
    
    return pred_meshes, merged_pred_mesh

def calculate_penetration_for_dpo_seed(pred_meshes, merged_pred_mesh, seed_dir):
    """
    Calculate penetration levels for a DPO seed
    
    Args:
        pred_meshes: List of individual predicted meshes
        merged_pred_mesh: Merged predicted mesh
        seed_dir: Seed directory path
        
    Returns:
        dict: Penetration level metrics
    """
    try:
        print(f"    Calculating penetration levels for seed: {seed_dir.name}")
        
        if not pred_meshes or merged_pred_mesh is None:
            print(f"    Skipping seed {seed_dir.name}: Missing meshes")
            return {}
        
        # Calculate penetration levels using messy kitchen specific functions
        penetration_metrics = {}
        
        # For predicted meshes
        print(f"      Computing penetration for predicted meshes...")
        pred_penetration = compute_penetration_level_detailed(pred_meshes, merged_pred_mesh, size=0.3, resolution=40)
        
        penetration_metrics = {
            'penetration_level': pred_penetration['penetration_level'],
            'overlap_ratio': pred_penetration['overlap_ratio'],
            'merged_internal_points': pred_penetration['merged_internal_points'],
            'individual_internal_points_sum': pred_penetration['individual_internal_points_sum'],
            'per_mesh_internal_points': pred_penetration['per_mesh_internal_points'],
            'per_mesh_penetration': pred_penetration['per_mesh_penetration'],
            'num_objects': len(pred_meshes)
        }
        
        print(f"        Penetration level: {penetration_metrics['penetration_level']:.6f}")
        print(f"        Overlap ratio: {penetration_metrics['overlap_ratio']:.6f}")
        print(f"        Number of objects: {penetration_metrics['num_objects']}")
        
        return penetration_metrics
        
    except Exception as e:
        print(f"      Error calculating penetration levels: {e}")
        return {}

def create_colored_merged_mesh(pred_meshes, seed_dir):
    """
    Create a colored version of the merged mesh with different colors for each object
    
    Args:
        pred_meshes: List of individual predicted meshes
        seed_dir: Seed directory path
    """
    try:
        if not pred_meshes:
            return
        
        print(f"      Creating colored merged mesh...")
        
        # Define 7 distinct colors (RGBA format)
        colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 1.0, 0.0, 1.0],  # Green
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0, 1.0],  # Orange
        ]
        
        # Create colored meshes
        colored_meshes = []
        for i, mesh in enumerate(pred_meshes):
            if i < len(colors):
                color = colors[i]
            else:
                # If we have more than 7 objects, cycle through colors
                color = colors[i % len(colors)]
            
            # Create a copy to avoid modifying original
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = color
            colored_meshes.append(mesh_copy)
        
        # Merge colored meshes
        if len(colored_meshes) > 1:
            colored_merged = trimesh.util.concatenate(colored_meshes)
        else:
            colored_merged = colored_meshes[0]
        
        # Save colored merged mesh
        colored_merged_file = seed_dir / "pred_merged_colored.ply"
        colored_merged.export(str(colored_merged_file))
        
        print(f"        Saved colored merged mesh: {colored_merged_file}")
        
        # Also save as GLB for better compatibility
        colored_merged_glb = seed_dir / "pred_merged_colored.glb"
        colored_merged.export(str(colored_merged_glb))
        
        print(f"        Saved colored merged mesh (GLB): {colored_merged_glb}")
        
    except Exception as e:
        print(f"        Error creating colored merged mesh: {e}")

def save_penetration_results_dpo(penetration_metrics, pred_meshes, seed_dir):
    """
    Save penetration analysis results to files in the seed directory
    
    Args:
        penetration_metrics: Dictionary containing penetration metrics
        pred_meshes: List of individual predicted meshes
        seed_dir: Seed directory path
    """
    try:
        # Save detailed results as text file
        detailed_file = seed_dir / "penetration_analysis.txt"
        with open(detailed_file, 'w') as f:
            f.write(f"Penetration Analysis for Seed: {seed_dir.name}\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"PENETRATION METRICS:\n")
            f.write(f"  Penetration Level: {penetration_metrics['penetration_level']:.6f}\n")
            f.write(f"  Overlap Ratio: {penetration_metrics['overlap_ratio']:.6f}\n")
            f.write(f"  Number of Objects: {penetration_metrics['num_objects']}\n")
            f.write(f"  Merged Internal Points: {penetration_metrics['merged_internal_points']}\n")
            f.write(f"  Individual Internal Points Sum: {penetration_metrics['individual_internal_points_sum']}\n")
            
            if 'per_mesh_internal_points' in penetration_metrics:
                f.write(f"  Per-mesh Internal Points: {penetration_metrics['per_mesh_internal_points']}\n")
            if 'per_mesh_penetration' in penetration_metrics:
                f.write(f"  Per-mesh Penetration: {[f'{p:.6f}' for p in penetration_metrics['per_mesh_penetration']]}\n")
        
        # Save summary JSON
        summary_file = seed_dir / "penetration_summary.json"
        
        # Convert numpy types to native Python types for JSON serialization
        json_metrics = {}
        for key, value in penetration_metrics.items():
            if isinstance(value, np.ndarray):
                json_metrics[key] = value.tolist()
            elif isinstance(value, np.integer):
                json_metrics[key] = int(value)
            elif isinstance(value, np.floating):
                json_metrics[key] = float(value)
            else:
                json_metrics[key] = value
        
        with open(summary_file, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        # Create colored merged mesh
        create_colored_merged_mesh(pred_meshes, seed_dir)
        
        print(f"      Penetration results saved to: {seed_dir}")
        
    except Exception as e:
        print(f"      Error saving penetration results: {e}")

def process_dpo_scene(scene_dir):
    """
    Process a single DPO scene directory
    
    Args:
        scene_dir (Path): Path to scene directory
    """
    print(f"\nProcessing DPO scene: {scene_dir.name}")
    
    # Get all seed subdirectories
    seed_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    print(f"  Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")
    
    scene_penetration_results = {}
    
    for seed_dir in seed_dirs:
        print(f"\n  Processing seed: {seed_dir.name}")
        
        # Load meshes for this seed
        pred_meshes, merged_pred_mesh = load_dpo_meshes_from_seed_dir(seed_dir)
        
        if not pred_meshes or merged_pred_mesh is None:
            print(f"    Skipping seed {seed_dir.name}: No valid meshes found")
            continue
        
        # Calculate penetration levels
        penetration_metrics = calculate_penetration_for_dpo_seed(pred_meshes, merged_pred_mesh, seed_dir)
        
        if penetration_metrics:
            # Save results to seed directory
            save_penetration_results_dpo(penetration_metrics, pred_meshes, seed_dir)
            
            # Store for scene-level summary
            scene_penetration_results[seed_dir.name] = penetration_metrics
        else:
            print(f"    Warning: Could not compute penetration metrics for seed {seed_dir.name}")
    
    # Save scene-level summary
    if scene_penetration_results:
        save_scene_summary(scene_penetration_results, scene_dir)
    
    return scene_penetration_results

def save_scene_summary(scene_penetration_results, scene_dir):
    """
    Save scene-level penetration summary
    
    Args:
        scene_penetration_results: Dictionary of seed results
        scene_dir: Scene directory path
    """
    try:
        # Calculate scene-level statistics
        penetration_levels = [metrics['penetration_level'] for metrics in scene_penetration_results.values()]
        overlap_ratios = [metrics['overlap_ratio'] for metrics in scene_penetration_results.values()]
        num_objects_list = [metrics['num_objects'] for metrics in scene_penetration_results.values()]
        
        scene_summary = {
            'scene_name': scene_dir.name,
            'num_seeds': len(scene_penetration_results),
            'penetration_levels': {
                'mean': float(np.mean(penetration_levels)),
                'std': float(np.std(penetration_levels)),
                'min': float(np.min(penetration_levels)),
                'max': float(np.max(penetration_levels)),
                'values': penetration_levels
            },
            'overlap_ratios': {
                'mean': float(np.mean(overlap_ratios)),
                'std': float(np.std(overlap_ratios)),
                'min': float(np.min(overlap_ratios)),
                'max': float(np.max(overlap_ratios)),
                'values': overlap_ratios
            },
            'num_objects': {
                'mean': float(np.mean(num_objects_list)),
                'std': float(np.std(num_objects_list)),
                'min': int(np.min(num_objects_list)),
                'max': int(np.max(num_objects_list)),
                'values': num_objects_list
            },
            'seed_details': scene_penetration_results
        }
        
        # Save scene summary
        scene_summary_file = scene_dir / "scene_penetration_summary.json"
        with open(scene_summary_file, 'w') as f:
            json.dump(scene_summary, f, indent=2)
        
        print(f"  Scene summary saved to: {scene_summary_file}")
        
        # Print scene summary
        print(f"\n  Scene {scene_dir.name} Summary:")
        print(f"    Seeds processed: {scene_summary['num_seeds']}")
        print(f"    Penetration Level: {scene_summary['penetration_levels']['mean']:.6f} ± {scene_summary['penetration_levels']['std']:.6f}")
        print(f"    Overlap Ratio: {scene_summary['overlap_ratios']['mean']:.6f} ± {scene_summary['overlap_ratios']['std']:.6f}")
        print(f"    Objects per seed: {scene_summary['num_objects']['mean']:.1f} ± {scene_summary['num_objects']['std']:.1f}")
        
    except Exception as e:
        print(f"  Error saving scene summary: {e}")

def evaluate_dpo_penetration(base_dir):
    """
    Evaluate penetration levels for all DPO scenes
    
    Args:
        base_dir (str): Base directory containing messy_kitchen_test_100 folder
    """
    base_path = Path(base_dir)
    dpo_data_path = base_path / "messy_kitchen_test_100"
    
    if not dpo_data_path.exists():
        print(f"Error: DPO data directory not found at {dpo_data_path}")
        return
    
    print(f"Processing DPO penetration analysis from: {dpo_data_path}")
    
    # Get all scene directories
    scene_dirs = [d for d in dpo_data_path.iterdir() if d.is_dir() and d.name.endswith('_combined')]
    print(f"Found {len(scene_dirs)} scene directories")
    
    all_scene_results = {}
    
    for scene_dir in scene_dirs:
        scene_results = process_dpo_scene(scene_dir)
        if scene_results:
            all_scene_results[scene_dir.name] = scene_results
    
    # Save overall summary
    if all_scene_results:
        save_overall_summary(all_scene_results, dpo_data_path)
    
    print(f"\nDPO penetration analysis completed for {len(all_scene_results)} scenes")

def save_overall_summary(all_scene_results, output_dir):
    """
    Save overall summary of all scenes
    
    Args:
        all_scene_results: Dictionary of all scene results
        output_dir: Output directory path
    """
    try:
        # Collect all penetration levels across all scenes and seeds
        all_penetration_levels = []
        all_overlap_ratios = []
        all_num_objects = []
        
        for scene_name, scene_results in all_scene_results.items():
            for seed_name, seed_metrics in scene_results.items():
                all_penetration_levels.append(seed_metrics['penetration_level'])
                all_overlap_ratios.append(seed_metrics['overlap_ratio'])
                all_num_objects.append(seed_metrics['num_objects'])
        
        overall_summary = {
            'total_scenes': len(all_scene_results),
            'total_seeds': sum(len(scene_results) for scene_results in all_scene_results.values()),
            'overall_penetration_levels': {
                'mean': float(np.mean(all_penetration_levels)),
                'std': float(np.std(all_penetration_levels)),
                'min': float(np.min(all_penetration_levels)),
                'max': float(np.max(all_penetration_levels))
            },
            'overall_overlap_ratios': {
                'mean': float(np.mean(all_overlap_ratios)),
                'std': float(np.std(all_overlap_ratios)),
                'min': float(np.min(all_overlap_ratios)),
                'max': float(np.max(all_overlap_ratios))
            },
            'overall_num_objects': {
                'mean': float(np.mean(all_num_objects)),
                'std': float(np.std(all_num_objects)),
                'min': int(np.min(all_num_objects)),
                'max': int(np.max(all_num_objects))
            },
            'scene_summaries': {}
        }
        
        # Add scene-level summaries
        for scene_name, scene_results in all_scene_results.items():
            scene_penetration_levels = [metrics['penetration_level'] for metrics in scene_results.values()]
            overall_summary['scene_summaries'][scene_name] = {
                'num_seeds': len(scene_results),
                'penetration_level_mean': float(np.mean(scene_penetration_levels)),
                'penetration_level_std': float(np.std(scene_penetration_levels))
            }
        
        # Save overall summary
        overall_summary_file = output_dir / "dpo_penetration_overall_summary.json"
        with open(overall_summary_file, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"\nOverall summary saved to: {overall_summary_file}")
        
        # Print overall summary
        print(f"\n{'='*60}")
        print(f"OVERALL DPO PENETRATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total scenes: {overall_summary['total_scenes']}")
        print(f"Total seeds: {overall_summary['total_seeds']}")
        print(f"Overall Penetration Level: {overall_summary['overall_penetration_levels']['mean']:.6f} ± {overall_summary['overall_penetration_levels']['std']:.6f}")
        print(f"Overall Overlap Ratio: {overall_summary['overall_overlap_ratios']['mean']:.6f} ± {overall_summary['overall_overlap_ratios']['std']:.6f}")
        print(f"Overall Objects per Seed: {overall_summary['overall_num_objects']['mean']:.1f} ± {overall_summary['overall_num_objects']['std']:.1f}")
        
    except Exception as e:
        print(f"Error saving overall summary: {e}")

if __name__ == "__main__":
    # Path to DPO data
    base_dir = "/home/ran.ding/projects/TARGO/dpo_data"
    
    print("DPO Penetration Analysis")
    print("="*50)
    
    # Evaluate DPO penetration
    evaluate_dpo_penetration(base_dir)
    
    print(f"\nDPO penetration analysis completed for: {base_dir}")
