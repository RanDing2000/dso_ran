#!/usr/bin/env python3
"""
Analyze GT objects from evaluation_objects_demo_test_100 directory.
Read all gt_object_*.ply files from each subfolder, concatenate them into merged meshes,
and provide statistics.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import trimesh
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will not be generated")


def find_gt_object_files(scene_dir: Path) -> List[Path]:
    """
    Find all gt_object_*.ply files in a scene directory.
    
    Args:
        scene_dir: Path to scene directory
        
    Returns:
        List of paths to gt_object_*.ply files
    """
    gt_files = []
    for file_path in scene_dir.glob("gt_object_*.ply"):
        gt_files.append(file_path)
    
    # Sort by object index for consistent ordering
    gt_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return gt_files


def load_and_concatenate_gt_objects(scene_dir: Path, verbose: bool = False) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Load all GT object files from a scene directory and concatenate them.
    
    Args:
        scene_dir: Path to scene directory
        verbose: Enable verbose output
        
    Returns:
        Tuple of (merged_mesh, statistics_dict)
    """
    gt_files = find_gt_object_files(scene_dir)
    
    if not gt_files:
        print(f"  No GT object files found in {scene_dir.name}")
        return None, {
            'scene_name': scene_dir.name,
            'n_objects': 0,
            'gt_files_found': [],
            'total_vertices': 0,
            'total_faces': 0,
            'total_volume': 0.0,
            'total_surface_area': 0.0,
            'bounds': None,
            'success': False
        }
    
    if verbose:
        print(f"  Found {len(gt_files)} GT object files:")
        for f in gt_files:
            print(f"    {f.name}")
    
    # Load individual meshes
    individual_meshes = []
    total_vertices = 0
    total_faces = 0
    total_volume = 0.0
    total_surface_area = 0.0
    
    for i, gt_file in enumerate(gt_files):
        try:
            mesh = trimesh.load(gt_file)
            
            # Check if mesh is valid (handle different trimesh versions)
            is_valid = True
            try:
                if hasattr(mesh, 'is_valid'):
                    is_valid = mesh.is_valid
                elif hasattr(mesh, 'is_valid'):
                    is_valid = mesh.is_valid
                else:
                    # Fallback: check if mesh has vertices and faces
                    is_valid = len(mesh.vertices) > 0 and len(mesh.faces) > 0
            except:
                is_valid = len(mesh.vertices) > 0 and len(mesh.faces) > 0
            
            if not is_valid:
                print(f"    Warning: Invalid mesh in {gt_file.name}")
                continue
            
            # Make mesh watertight if possible (handle different trimesh versions)
            try:
                if hasattr(mesh, 'is_watertight') and not mesh.is_watertight:
                    if hasattr(mesh, 'fill_holes'):
                        mesh.fill_holes()
                    if hasattr(mesh, 'remove_degenerate_faces'):
                        mesh.remove_degenerate_faces()
                    if hasattr(mesh, 'remove_duplicate_faces'):
                        mesh.remove_duplicate_faces()
                    if hasattr(mesh, 'remove_infinite_values'):
                        mesh.remove_infinite_values()
                    if hasattr(mesh, 'fix_normals'):
                        mesh.fix_normals()
            except Exception as e:
                print(f"    Warning: Could not fix mesh {gt_file.name}: {e}")
            
            individual_meshes.append(mesh)
            
            # Accumulate statistics
            total_vertices += len(mesh.vertices)
            total_faces += len(mesh.faces)
            total_volume += mesh.volume if hasattr(mesh, 'volume') else 0.0
            total_surface_area += mesh.area if hasattr(mesh, 'area') else 0.0
            
            if verbose:
                print(f"    Loaded {gt_file.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                
        except Exception as e:
            print(f"    Error loading {gt_file.name}: {e}")
            continue
    
    if not individual_meshes:
        print(f"  No valid meshes loaded from {scene_dir.name}")
        return None, {
            'scene_name': scene_dir.name,
            'n_objects': 0,
            'gt_files_found': [f.name for f in gt_files],
            'total_vertices': 0,
            'total_faces': 0,
            'total_volume': 0.0,
            'total_surface_area': 0.0,
            'bounds': None,
            'success': False
        }
    
    # Concatenate all meshes
    try:
        if len(individual_meshes) == 1:
            merged_mesh = individual_meshes[0]
        else:
            merged_mesh = trimesh.util.concatenate(individual_meshes)
        
        # Calculate bounds of merged mesh
        bounds = merged_mesh.bounds if merged_mesh is not None else None
        
        if verbose:
            print(f"  Successfully concatenated {len(individual_meshes)} meshes")
            if merged_mesh is not None:
                print(f"    Merged mesh: {len(merged_mesh.vertices)} vertices, {len(merged_mesh.faces)} faces")
        
        statistics = {
            'scene_name': scene_dir.name,
            'n_objects': len(individual_meshes),
            'gt_files_found': [f.name for f in gt_files],
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'total_volume': total_volume,
            'total_surface_area': total_surface_area,
            'merged_vertices': len(merged_mesh.vertices) if merged_mesh else 0,
            'merged_faces': len(merged_mesh.faces) if merged_mesh else 0,
            'merged_volume': merged_mesh.volume if merged_mesh and hasattr(merged_mesh, 'volume') else 0.0,
            'merged_surface_area': merged_mesh.area if merged_mesh and hasattr(merged_mesh, 'area') else 0.0,
            'bounds': bounds.tolist() if bounds is not None else None,
            'success': True
        }
        
        return merged_mesh, statistics
        
    except Exception as e:
        print(f"  Error concatenating meshes in {scene_dir.name}: {e}")
        return None, {
            'scene_name': scene_dir.name,
            'n_objects': len(individual_meshes),
            'gt_files_found': [f.name for f in gt_files],
            'total_vertices': total_vertices,
            'total_faces': total_faces,
            'total_volume': total_volume,
            'total_surface_area': total_surface_area,
            'merged_vertices': 0,
            'merged_faces': 0,
            'merged_volume': 0.0,
            'merged_surface_area': 0.0,
            'bounds': None,
            'success': False,
            'error': str(e)
        }


def analyze_all_scenes(base_dir: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Analyze all scenes in the base directory.
    
    Args:
        base_dir: Base directory containing scene subdirectories
        verbose: Enable verbose output
        
    Returns:
        List of statistics dictionaries for each scene
    """
    scene_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.endswith('_combined')]
    scene_dirs.sort()
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    all_statistics = []
    success_count = 0
    fail_count = 0
    
    for i, scene_dir in enumerate(scene_dirs):
        print(f"Processing scene {i+1}/{len(scene_dirs)}: {scene_dir.name}")
        
        try:
            merged_mesh, statistics = load_and_concatenate_gt_objects(scene_dir, verbose=verbose)
            
            if statistics['success']:
                success_count += 1
                print(f"  ✓ Success: {statistics['n_objects']} objects, {statistics['merged_vertices']} vertices, {statistics['merged_faces']} faces")
            else:
                fail_count += 1
                print(f"  ✗ Failed: {statistics.get('error', 'Unknown error')}")
            
            all_statistics.append(statistics)
            
        except Exception as e:
            print(f"  ✗ Error processing {scene_dir.name}: {e}")
            fail_count += 1
            all_statistics.append({
                'scene_name': scene_dir.name,
                'n_objects': -1,
                'gt_files_found': [],
                'total_vertices': -1,
                'total_faces': -1,
                'total_volume': -1.0,
                'total_surface_area': -1.0,
                'merged_vertices': -1,
                'merged_faces': -1,
                'merged_volume': -1.0,
                'merged_surface_area': -1.0,
                'bounds': None,
                'success': False,
                'error': str(e)
            })
    
    print(f"\nProcessing complete: {success_count} successful, {fail_count} failed")
    return all_statistics


def save_statistics(statistics: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Save statistics to JSON and CSV files.
    
    Args:
        statistics: List of statistics dictionaries
        output_dir: Output directory for saving files
    """
    output_dir.mkdir(exist_ok=True)
    
    # Save as JSON
    json_file = output_dir / "gt_objects_statistics.json"
    with open(json_file, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved statistics to {json_file}")
    
    # Save as CSV
    csv_file = output_dir / "gt_objects_statistics.csv"
    if statistics:
        import csv
        keys = [
            'scene_name', 'n_objects', 'total_vertices', 'total_faces', 
            'total_volume', 'total_surface_area', 'merged_vertices', 
            'merged_faces', 'merged_volume', 'merged_surface_area', 'success'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for stat in statistics:
                writer.writerow({k: stat.get(k, '') for k in keys})
        
        print(f"Saved CSV to {csv_file}")


def plot_statistics(statistics: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    Create plots for the statistics.
    
    Args:
        statistics: List of statistics dictionaries
        output_dir: Output directory for saving plots
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not available)")
        return
    
    # Filter successful cases
    successful_stats = [s for s in statistics if s.get('success', False)]
    
    if not successful_stats:
        print("No successful cases to plot")
        return
    
    print(f"Creating plots for {len(successful_stats)} successful cases...")
    
    # Extract data for plotting
    n_objects = [s['n_objects'] for s in successful_stats]
    total_vertices = [s['total_vertices'] for s in successful_stats]
    total_faces = [s['total_faces'] for s in successful_stats]
    total_volumes = [s['total_volume'] for s in successful_stats]
    total_surface_areas = [s['total_surface_area'] for s in successful_stats]
    merged_vertices = [s['merged_vertices'] for s in successful_stats]
    merged_faces = [s['merged_faces'] for s in successful_stats]
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'GT Objects Statistics (Total: {len(successful_stats)} scenes)', fontsize=16)
    
    # Plot 1: Number of objects per scene
    axes[0, 0].hist(n_objects, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'N Objects per Scene\nMean: {np.mean(n_objects):.1f} ± {np.std(n_objects):.1f}')
    axes[0, 0].set_xlabel('Number of Objects')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total vertices per scene
    axes[0, 1].hist(total_vertices, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title(f'Total Vertices per Scene\nMean: {np.mean(total_vertices):.0f} ± {np.std(total_vertices):.0f}')
    axes[0, 1].set_xlabel('Total Vertices')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Total faces per scene
    axes[0, 2].hist(total_faces, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title(f'Total Faces per Scene\nMean: {np.mean(total_faces):.0f} ± {np.std(total_faces):.0f}')
    axes[0, 2].set_xlabel('Total Faces')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Total volume per scene
    axes[1, 0].hist(total_volumes, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_title(f'Total Volume per Scene\nMean: {np.mean(total_volumes):.6f} ± {np.std(total_volumes):.6f}')
    axes[1, 0].set_xlabel('Total Volume')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Total surface area per scene
    axes[1, 1].hist(total_surface_areas, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title(f'Total Surface Area per Scene\nMean: {np.mean(total_surface_areas):.6f} ± {np.std(total_surface_areas):.6f}')
    axes[1, 1].set_xlabel('Total Surface Area')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Merged vs Individual vertices comparison
    axes[1, 2].scatter(total_vertices, merged_vertices, alpha=0.6)
    axes[1, 2].plot([0, max(total_vertices)], [0, max(total_vertices)], 'r--', alpha=0.8, label='y=x')
    axes[1, 2].set_title('Merged vs Individual Vertices')
    axes[1, 2].set_xlabel('Sum of Individual Vertices')
    axes[1, 2].set_ylabel('Merged Mesh Vertices')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "gt_objects_statistics_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plots to {plot_file}")


def print_summary(statistics: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics.
    
    Args:
        statistics: List of statistics dictionaries
    """
    successful_stats = [s for s in statistics if s.get('success', False)]
    failed_stats = [s for s in statistics if not s.get('success', False)]
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total scenes processed: {len(statistics)}")
    print(f"Successful scenes: {len(successful_stats)}")
    print(f"Failed scenes: {len(failed_stats)}")
    
    if successful_stats:
        print(f"\nSuccessful scenes statistics:")
        n_objects = [s['n_objects'] for s in successful_stats]
        total_vertices = [s['total_vertices'] for s in successful_stats]
        total_faces = [s['total_faces'] for s in successful_stats]
        total_volumes = [s['total_volume'] for s in successful_stats]
        total_surface_areas = [s['total_surface_area'] for s in successful_stats]
        
        print(f"  Objects per scene: {np.mean(n_objects):.2f} ± {np.std(n_objects):.2f} (range: {min(n_objects)}-{max(n_objects)})")
        print(f"  Total vertices per scene: {np.mean(total_vertices):.0f} ± {np.std(total_vertices):.0f}")
        print(f"  Total faces per scene: {np.mean(total_faces):.0f} ± {np.std(total_faces):.0f}")
        print(f"  Total volume per scene: {np.mean(total_volumes):.6f} ± {np.std(total_volumes):.6f}")
        print(f"  Total surface area per scene: {np.mean(total_surface_areas):.6f} ± {np.std(total_surface_areas):.6f}")
        
        # Check for efficiency (merged vs individual)
        merged_vertices = [s['merged_vertices'] for s in successful_stats]
        vertex_efficiency = [m/t if t > 0 else 0 for m, t in zip(merged_vertices, total_vertices)]
        print(f"  Vertex efficiency (merged/individual): {np.mean(vertex_efficiency):.3f} ± {np.std(vertex_efficiency):.3f}")
    
    if failed_stats:
        print(f"\nFailed scenes:")
        for stat in failed_stats:
            print(f"  {stat['scene_name']}: {stat.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze GT objects from evaluation_objects_demo_test_100 directory")
    parser.add_argument("--base_dir", type=str, 
                       default="messy_kitchen_results/evaluation_objects_demo_test_100",
                       help="Base directory containing scene subdirectories")
    parser.add_argument("--output_dir", type=str, 
                       default="messy_kitchen_results/gt_objects_analysis",
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        sys.exit(1)
    
    print(f"Analyzing GT objects in: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Verbose mode: {args.verbose}")
    
    start_time = time.time()
    
    # Analyze all scenes
    statistics = analyze_all_scenes(base_dir, verbose=args.verbose)
    
    # Save results
    save_statistics(statistics, output_dir)
    
    # Create plots
    plot_statistics(statistics, output_dir)
    
    # Print summary
    print_summary(statistics)
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
