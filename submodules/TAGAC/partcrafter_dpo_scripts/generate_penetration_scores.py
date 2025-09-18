#!/usr/bin/env python3
"""
Generate penetration scores for DPO data directories
Traverses messy_kitchen_data/dpo_data/messy_kitchen_configs and generates penetration_score.json for each scene
"""

import os
import json
import sys
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict, Any
import argparse
from utils_giga import normalize_dpo_mesh, apply_dpo_transform

# Add TAGAC utils to path
sys.path.append('/home/ran.ding/messy-kitchen/dso/submodules/TAGAC/src')
from utils_giga import compute_penetration_level_detailed

def load_meshes_from_seed_dir(seed_dir: Path) -> tuple:
    """
    Load meshes from a seed directory (e.g., 0/, 1/, 2/, etc.)
    
    Args:
        seed_dir: Path to seed directory containing pred_merged.glb and pred_obj_*.ply
        
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
                mesh = trimesh.load(str(pred_file))
                pred_meshes.append(mesh)
                print(f"    Loaded individual mesh: {pred_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_file.name}: {e}")
        
        # Load merged predicted mesh (pred_merged.glb)
        pred_merged_file = seed_dir / "pred_merged.glb"
        if pred_merged_file.exists():
            try:
                merged_pred_mesh = trimesh.load(str(pred_merged_file))
                print(f"    Loaded merged mesh: {pred_merged_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_merged_file.name}: {e}")
        else:
            print(f"    Warning: Merged mesh file not found: {pred_merged_file}")
            # Create merged mesh from individual meshes if available
            if pred_meshes:
                try:
                    merged_pred_mesh = trimesh.util.concatenate(pred_meshes)
                    print(f"    Created merged mesh from {len(pred_meshes)} individual meshes")
                except Exception as e:
                    print(f"    Error creating merged mesh: {e}")
        
        print(f"    Scene {seed_dir.parent.name}/{seed_dir.name}: {len(pred_meshes)} individual meshes, merged: {merged_pred_mesh is not None}")
        
    except Exception as e:
        print(f"    Error processing seed directory {seed_dir}: {e}")
    
    return pred_meshes, merged_pred_mesh

def calculate_penetration_score(pred_meshes: List[trimesh.Trimesh], merged_pred_mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Calculate penetration score using TAGAC's compute_penetration_level_detailed
    
    Args:
        pred_meshes: List of individual predicted meshes
        merged_pred_mesh: Merged predicted mesh
        
    Returns:
        Dict containing penetration metrics
    """
    try:
        if not pred_meshes or merged_pred_mesh is None:
            print(f"      Warning: Insufficient meshes for penetration calculation")
            return {
                'penetration_level': 0.0,
                'overlap_ratio': 0.0,
                'merged_internal_points': 0,
                'individual_internal_points_sum': 0,
                'per_mesh_internal_points': [],
                'per_mesh_penetration': [],
                'error': 'Insufficient meshes'
            }
        
        print(f"Computing penetration for {len(pred_meshes)} individual meshes...")

        pred_merged_mesh = trimesh.util.concatenate(pred_meshes)
        print("Bounding box extents:", pred_merged_mesh.extents)  # (x_size, y_size, z_size)
        print("Bounding box bounds:\n", pred_merged_mesh.bounds)

        normalized_pred_merged_mesh, shift, scale = normalize_dpo_mesh(pred_merged_mesh)
        normalized_pred_meshes = []
        for pred_mesh in pred_meshes:
            # normalized_pred_mesh, shift, scale = normalize_dpo_mesh(pred_mesh)
            normalized_pred_mesh = apply_dpo_transform(pred_mesh, shift, scale)
            normalized_pred_meshes.append(normalized_pred_mesh)
        
        # # save normalized_pred_meshes to ply files
        # for i, normalized_pred_mesh in enumerate(normalized_pred_meshes):
        #     normalized_pred_mesh.export(f"tmp3/normalized_pred_mesh_{i}.ply")
        # normalized_pred_merged_mesh.export(f"tmp3/normalized_pred_merged_mesh.ply")

        
        # Use TAGAC's penetration calculation
        penetration_metrics = compute_penetration_level_detailed(
            mesh_list=normalized_pred_meshes,
            # merged_mesh=merged_pred_mesh,
            merged_mesh=normalized_pred_merged_mesh,
            size=0.3,
            resolution=40
        )
        
        # Convert numpy types to native Python types for JSON serialization
        result = {
            'penetration_level': float(penetration_metrics['penetration_level']),
            'overlap_ratio': float(penetration_metrics['overlap_ratio']),
            'merged_internal_points': int(penetration_metrics['merged_internal_points']),
            'individual_internal_points_sum': int(penetration_metrics['individual_internal_points_sum']),
            'per_mesh_internal_points': [int(x) for x in penetration_metrics['per_mesh_internal_points']],
            'per_mesh_penetration': [float(x) for x in penetration_metrics['per_mesh_penetration']]
        }
        
        print(f"      Penetration level: {result['penetration_level']:.6f}")
        print(f"      Overlap ratio: {result['overlap_ratio']:.6f}")
        print(f"      Per-mesh penetrations: {[f'{p:.6f}' for p in result['per_mesh_penetration']]}")
        
        return result
        
    except Exception as e:
        print(f"      Error calculating penetration score: {e}")
        return {
            'penetration_level': 0.0,
            'overlap_ratio': 0.0,
            'merged_internal_points': 0,
            'individual_internal_points_sum': 0,
            'per_mesh_internal_points': [],
            'per_mesh_penetration': [],
            'error': str(e)
        }

def process_scene_directory(scene_dir: Path) -> Dict[str, Any]:
    """
    Process a single scene directory containing multiple seed directories
    
    Args:
        scene_dir: Path to scene directory (e.g., 0b4113d2beec4002989365364b4e37e5_combined/)
        
    Returns:
        Dict containing penetration scores for all seeds
    """
    print(f"\nProcessing scene: {scene_dir.name}")
    
    scene_results = {
        'scene_name': scene_dir.name,
        'seeds': {},
        'summary': {}
    }
    
    # Get all seed directories (0/, 1/, 2/, etc.)
    seed_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    seed_dirs = sorted(seed_dirs, key=lambda x: int(x.name))
    
    print(f"  Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")
    
    penetration_scores = []
    
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        print(f"  Processing seed: {seed}")
        
        # Load meshes for this seed
        pred_meshes, merged_pred_mesh = load_meshes_from_seed_dir(seed_dir)
        
        # Calculate penetration score
        penetration_result = calculate_penetration_score(pred_meshes, merged_pred_mesh)
        
        # Save individual penetration score for this seed
        seed_penetration_file = seed_dir / "penetration_score.json"
        seed_data = {
            'seed': int(seed),
            'scene_name': scene_dir.name,
            'num_individual_meshes': len(pred_meshes),
            'has_merged_mesh': merged_pred_mesh is not None,
            'penetration_score': penetration_result,
            'timestamp': str(Path().cwd())  # Simple timestamp placeholder
        }
        
        with open(seed_penetration_file, 'w') as f:
            json.dump(seed_data, f, indent=2)
        print(f"    Saved seed penetration score: {seed_penetration_file}")
        
        # Store results for this seed
        scene_results['seeds'][seed] = {
            'seed': int(seed),
            'num_individual_meshes': len(pred_meshes),
            'has_merged_mesh': merged_pred_mesh is not None,
            'penetration_score': penetration_result
        }
        
        if 'error' not in penetration_result:
            penetration_scores.append(penetration_result['penetration_level'])
    
    # Calculate summary statistics
    if penetration_scores:
        scene_results['summary'] = {
            'total_seeds': len(seed_dirs),
            'valid_penetration_scores': len(penetration_scores),
            'avg_penetration_level': float(np.mean(penetration_scores)),
            'std_penetration_level': float(np.std(penetration_scores)),
            'min_penetration_level': float(np.min(penetration_scores)),
            'max_penetration_level': float(np.max(penetration_scores)),
            'penetration_levels': [float(x) for x in penetration_scores]
        }
        
        print(f"  Scene summary:")
        print(f"    Total seeds: {scene_results['summary']['total_seeds']}")
        print(f"    Valid penetration scores: {scene_results['summary']['valid_penetration_scores']}")
        print(f"    Average penetration: {scene_results['summary']['avg_penetration_level']:.6f} ± {scene_results['summary']['std_penetration_level']:.6f}")
        print(f"    Range: [{scene_results['summary']['min_penetration_level']:.6f}, {scene_results['summary']['max_penetration_level']:.6f}]")
    else:
        scene_results['summary'] = {
            'total_seeds': len(seed_dirs),
            'valid_penetration_scores': 0,
            'error': 'No valid penetration scores computed'
        }
        print(f"  Warning: No valid penetration scores computed for scene {scene_dir.name}")
    
    return scene_results

def generate_penetration_scores(base_dir: str, output_file: str = None):
    """
    Generate penetration scores for all scenes in the DPO data directory
    
    Args:
        base_dir: Base directory containing scene directories
        output_file: Optional output file path for consolidated results
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Base directory not found: {base_path}")
        return
    
    print(f"Generating penetration scores from: {base_path}")
    
    # Get all scene directories
    scene_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith('_combined')]
    scene_dirs = sorted(scene_dirs)
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    all_results = {
        'base_directory': str(base_path),
        'total_scenes': len(scene_dirs),
        'scenes': {},
        'overall_summary': {}
    }
    
    all_penetration_scores = []
    valid_scenes = 0
    
    for scene_dir in scene_dirs:
        try:
            # Process this scene
            scene_results = process_scene_directory(scene_dir)
            
            # Store in overall results
            all_results['scenes'][scene_dir.name] = scene_results
            
            # Collect penetration scores for overall summary
            if scene_results['summary']['valid_penetration_scores'] > 0:
                all_penetration_scores.extend(scene_results['summary']['penetration_levels'])
                valid_scenes += 1
                
                # Save individual scene penetration score
                scene_penetration_file = scene_dir / "penetration_score.json"
                with open(scene_penetration_file, 'w') as f:
                    json.dump(scene_results, f, indent=2)
                print(f"  Saved scene penetration score: {scene_penetration_file}")
            
        except Exception as e:
            print(f"  Error processing scene {scene_dir.name}: {e}")
            continue
    
    # Calculate overall summary
    if all_penetration_scores:
        all_results['overall_summary'] = {
            'total_scenes': len(scene_dirs),
            'valid_scenes': valid_scenes,
            'total_penetration_scores': len(all_penetration_scores),
            'overall_avg_penetration': float(np.mean(all_penetration_scores)),
            'overall_std_penetration': float(np.std(all_penetration_scores)),
            'overall_min_penetration': float(np.min(all_penetration_scores)),
            'overall_max_penetration': float(np.max(all_penetration_scores))
        }
        
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Total scenes: {all_results['overall_summary']['total_scenes']}")
        print(f"Valid scenes: {all_results['overall_summary']['valid_scenes']}")
        print(f"Total penetration scores: {all_results['overall_summary']['total_penetration_scores']}")
        print(f"Overall average penetration: {all_results['overall_summary']['overall_avg_penetration']:.6f} ± {all_results['overall_summary']['overall_std_penetration']:.6f}")
        print(f"Overall range: [{all_results['overall_summary']['overall_min_penetration']:.6f}, {all_results['overall_summary']['overall_max_penetration']:.6f}]")
    else:
        all_results['overall_summary'] = {
            'total_scenes': len(scene_dirs),
            'valid_scenes': 0,
            'error': 'No valid penetration scores computed'
        }
        print(f"\nWarning: No valid penetration scores computed for any scenes")
    
    # Save consolidated results if output file specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nConsolidated results saved to: {output_path}")
    
    print(f"\nPenetration score generation completed!")
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Generate penetration scores for DPO data')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs',
                       help='Base directory containing scene directories')
    parser.add_argument('--scene_root_dir', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs',
                       help='Root directory containing multiple scene directories (each ending with _combined)')
    parser.add_argument('--scene_dir', type=str, 
                       default=None,
                       help='Specific scene directory to process (overrides scene_root_dir)')
    parser.add_argument('--output_file', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/penetration_scores_consolidated.json',
                       help='Output file for consolidated results')
    
    args = parser.parse_args()
    
    # Check if specific scene directory is provided
    if args.scene_dir:
        scene_path = Path(args.scene_dir)
        if scene_path.exists() and scene_path.is_dir():
            print(f"Processing specific scene directory: {scene_path}")
            results = process_scene_directory(scene_path)
            
            # Save scene-level results
            scene_output_file = scene_path / "scene_penetration_summary.json"
            with open(scene_output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nScene-level results saved to: {scene_output_file}")
        else:
            print(f"Error: Scene directory not found: {scene_path}")
            return
    else:
        # Process all scene directories in scene_root_dir
        scene_root_path = Path(args.scene_root_dir)
        if not scene_root_path.exists():
            print(f"Error: Scene root directory not found: {scene_root_path}")
            return
        
        print(f"Processing all scene directories in: {scene_root_path}")
        
        # Get all scene directories (ending with _combined)
        scene_dirs = [d for d in scene_root_path.iterdir() if d.is_dir() and d.name.endswith('_combined')]
        scene_dirs = sorted(scene_dirs)
        
        print(f"Found {len(scene_dirs)} scene directories: {[d.name for d in scene_dirs]}")
        
        all_results = {
            'scene_root_directory': str(scene_root_path),
            'total_scenes': len(scene_dirs),
            'scenes': {},
            'overall_summary': {}
        }
        
        all_penetration_scores = []
        valid_scenes = 0
        
        for scene_dir in scene_dirs:
            try:
                print(f"\n{'='*80}")
                print(f"Processing scene: {scene_dir.name}")
                print(f"{'='*80}")
                
                # Process this scene directory
                scene_results = process_scene_directory(scene_dir)
                
                # Store in overall results
                all_results['scenes'][scene_dir.name] = scene_results
                
                # Collect penetration scores for overall summary
                if scene_results['summary']['valid_penetration_scores'] > 0:
                    all_penetration_scores.extend(scene_results['summary']['penetration_levels'])
                    valid_scenes += 1
                
            except Exception as e:
                print(f"  Error processing scene {scene_dir.name}: {e}")
                continue
        
        # Calculate overall summary
        if all_penetration_scores:
            all_results['overall_summary'] = {
                'total_scenes': len(scene_dirs),
                'valid_scenes': valid_scenes,
                'total_penetration_scores': len(all_penetration_scores),
                'overall_avg_penetration': float(np.mean(all_penetration_scores)),
                'overall_std_penetration': float(np.std(all_penetration_scores)),
                'overall_min_penetration': float(np.min(all_penetration_scores)),
                'overall_max_penetration': float(np.max(all_penetration_scores))
            }
            
            print(f"\n{'='*80}")
            print(f"OVERALL SUMMARY")
            print(f"{'='*80}")
            print(f"Total scenes: {all_results['overall_summary']['total_scenes']}")
            print(f"Valid scenes: {all_results['overall_summary']['valid_scenes']}")
            print(f"Total penetration scores: {all_results['overall_summary']['total_penetration_scores']}")
            print(f"Overall average penetration: {all_results['overall_summary']['overall_avg_penetration']:.6f} ± {all_results['overall_summary']['overall_std_penetration']:.6f}")
            print(f"Overall range: [{all_results['overall_summary']['overall_min_penetration']:.6f}, {all_results['overall_summary']['overall_max_penetration']:.6f}]")
        else:
            all_results['overall_summary'] = {
                'total_scenes': len(scene_dirs),
                'valid_scenes': 0,
                'error': 'No valid penetration scores computed'
            }
            print(f"\nWarning: No valid penetration scores computed for any scenes")
        
        # Save consolidated results
        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\nConsolidated results saved to: {output_path}")
    
    print(f"\nProcessing completed!")

if __name__ == "__main__":
    main()
