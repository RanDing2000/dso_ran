#!/usr/bin/env python3
"""
Demo script for testing penetration scores between two objects as they gradually approach each other.

This script loads two PLY files (big-bowl.ply and cup-plate.ply) and simulates 
5 steps of gradual approach to test penetration scoring functionality.

Features:
- High-resolution TSDF (80x80x80 = 512K voxels) for accurate penetration calculation
- Scene normalization to [0, 0.3] range for consistent measurements
- Step-by-step visualization and result saving
"""

import os
import sys
import numpy as np
import trimesh
from pathlib import Path

# Add the src directory to the path to import utils_giga
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from utils_giga import (
    compute_penetration_level,
    compute_penetration_level_detailed,
    compute_bounding_box_iou
)

def load_demo_meshes():
    """
    Load the two demo meshes for penetration testing
    
    Returns:
        tuple: (cup_mesh, bowl_mesh) - The loaded meshes
    """
    # Define paths to the demo meshes
    base_dir = Path("/home/ran.ding/projects/TARGO/data/messy_kitchen_real/demo-sc-1/scene-split")
    cup_path = base_dir / "big-bowl.ply"
    bowl_path = base_dir / "cup-plate.ply"
    
    print(f"Loading demo meshes from: {base_dir}")
    
    # Check if files exist
    if not cup_path.exists():
        raise FileNotFoundError(f"Cup mesh not found: {cup_path}")
    if not bowl_path.exists():
        raise FileNotFoundError(f"Bowl mesh not found: {bowl_path}")
    
    # Load meshes
    try:
        cup_mesh = trimesh.load_mesh(str(cup_path))
        bowl_mesh = trimesh.load_mesh(str(bowl_path))
        
        print(f"✓ Loaded big-bowl mesh: {cup_path.name}")
        print(f"  - Vertices: {len(cup_mesh.vertices)}")
        print(f"  - Faces: {len(cup_mesh.faces)}")
        print(f"  - Bounds: {cup_mesh.bounds}")
        print(f"  - Is watertight: {cup_mesh.is_watertight}")
        
        print(f"✓ Loaded cup-plate mesh: {bowl_path.name}")
        print(f"  - Vertices: {len(bowl_mesh.vertices)}")
        print(f"  - Faces: {len(bowl_mesh.faces)}")
        print(f"  - Bounds: {bowl_mesh.bounds}")
        print(f"  - Is watertight: {bowl_mesh.is_watertight}")
        
        return cup_mesh, bowl_mesh
        
    except Exception as e:
        raise RuntimeError(f"Error loading meshes: {e}")

def normalize_mesh_to_scene(mesh, scene_bounds, target_size=0.3):
    """
    Normalize a mesh to fit within the scene bounds [0, target_size]
    
    Args:
        mesh: Input mesh
        scene_bounds: Overall scene bounds [min_coords, max_coords]
        target_size: Target scene size (default: 0.3)
        
    Returns:
        Normalized mesh
    """
    # Make a copy
    normalized_mesh = mesh.copy()
    
    # Get scene dimensions
    scene_min = scene_bounds[0]
    scene_max = scene_bounds[1]
    scene_dimensions = scene_max - scene_min
    scene_center = (scene_min + scene_max) / 2
    
    # First translate to center scene at origin
    normalized_mesh.apply_translation(-scene_center)
    
    # Then scale to fit within [0, target_size]
    max_scene_dimension = np.max(scene_dimensions)
    if max_scene_dimension > 0:
        scale_factor = target_size / max_scene_dimension
        normalized_mesh.apply_scale(scale_factor)
    
    # Finally translate to have minimum at 0 and center around target_size/2
    normalized_mesh.apply_translation([target_size/2, target_size/2, target_size/2])
    
    return normalized_mesh, scale_factor

def get_combined_scene_bounds(meshes):
    """
    Get the combined bounding box of multiple meshes
    
    Args:
        meshes: List of meshes
        
    Returns:
        Combined bounds [min_coords, max_coords]
    """
    if not meshes:
        return np.array([[0, 0, 0], [1, 1, 1]])
    
    all_bounds = [mesh.bounds for mesh in meshes]
    min_coords = np.min([bounds[0] for bounds in all_bounds], axis=0)
    max_coords = np.max([bounds[1] for bounds in all_bounds], axis=0)
    
    return np.array([min_coords, max_coords])

def prepare_meshes_for_demo(cup_mesh, bowl_mesh):
    """
    Prepare meshes for the penetration demo by normalizing the entire scene to [0, 0.3] range
    
    Args:
        cup_mesh: The big-bowl mesh
        bowl_mesh: The cup-plate mesh
        
    Returns:
        tuple: (prepared_cup, prepared_bowl, initial_separation, scale_factor)
    """
    print("\nPreparing meshes for demo...")
    
    # Get combined scene bounds before normalization
    print("Calculating combined scene bounds...")
    combined_bounds = get_combined_scene_bounds([cup_mesh, bowl_mesh])
    scene_dimensions = combined_bounds[1] - combined_bounds[0]
    scene_center = (combined_bounds[0] + combined_bounds[1]) / 2
    
    print(f"Original scene bounds: {combined_bounds}")
    print(f"Original scene dimensions: {scene_dimensions}")
    print(f"Original scene center: {scene_center}")
    
    # Normalize both meshes to the same scene coordinate system [0, 0.3]
    print("\nNormalizing entire scene to [0, 0.3] range...")
    cup, scale_factor = normalize_mesh_to_scene(cup_mesh, combined_bounds, target_size=0.3)
    bowl, _ = normalize_mesh_to_scene(bowl_mesh, combined_bounds, target_size=0.3)
    
    print(f"Scene scale factor: {scale_factor:.6f}")
    
    # Get normalized bounds and dimensions
    cup_bounds = cup.bounds
    bowl_bounds = bowl.bounds
    
    cup_center = (cup_bounds[0] + cup_bounds[1]) / 2
    bowl_center = (bowl_bounds[0] + bowl_bounds[1]) / 2
    
    cup_dims = cup_bounds[1] - cup_bounds[0]
    bowl_dims = bowl_bounds[1] - bowl_bounds[0]
    
    print(f"\nAfter scene normalization:")
    print(f"Big-bowl bounds: {cup_bounds}")
    print(f"Big-bowl center: {cup_center}")
    print(f"Big-bowl dimensions: {cup_dims}")
    print(f"Cup-plate bounds: {bowl_bounds}")
    print(f"Cup-plate center: {bowl_center}")
    print(f"Cup-plate dimensions: {bowl_dims}")
    
    # Position cup-plate at a fixed location within the scene
    target_bowl_center = np.array([0.15, 0.15, 0.15])  # Center of [0, 0.3] space
    bowl_translation = target_bowl_center - bowl_center
    bowl.apply_translation(bowl_translation)
    bowl_center_new = (bowl.bounds[0] + bowl.bounds[1]) / 2
    print(f"Cup-plate moved to: {bowl_center_new}")
    
    # Position nezo at initial separation distance
    safety_margin = 0.05  # 5cm safety margin in the 0.3 space
    initial_separation_distance = max(cup_dims[0], bowl_dims[0]) / 2 + max(cup_dims[0], bowl_dims[0]) / 2 + safety_margin
    
    # Move big-bowl to the right of cup-plate
    target_cup_x = bowl_center_new[0] + initial_separation_distance
    target_cup_center = np.array([target_cup_x, bowl_center_new[1], bowl_center_new[2]])
    cup_translation = target_cup_center - cup_center
    cup.apply_translation(cup_translation)
    cup_center_new = (cup.bounds[0] + cup.bounds[1]) / 2
    
    print(f"Big-bowl moved to: {cup_center_new}")
    print(f"Initial separation distance: {initial_separation_distance:.4f}")
    
    # Verify both objects are within [0, 0.3] bounds
    final_combined_bounds = get_combined_scene_bounds([cup, bowl])
    print(f"Final scene bounds: {final_combined_bounds}")
    print(f"Scene fits in [0, 0.3]: {np.all(final_combined_bounds[0] >= -0.001) and np.all(final_combined_bounds[1] <= 0.301)}")
    
    return cup, bowl, initial_separation_distance, scale_factor

def create_demo_steps(cup_mesh, bowl_mesh, initial_separation, num_steps=5):
    """
    Create demo steps with gradually decreasing separation
    
    Args:
        cup_mesh: The cup mesh positioned at initial separation
        bowl_mesh: The bowl mesh at origin
        initial_separation: Initial separation distance
        num_steps: Number of demo steps
        
    Returns:
        list: List of (cup_mesh, bowl_mesh, separation_distance, step_description) tuples
    """
    print(f"\nCreating {num_steps} demo steps...")
    
    steps = []
    
    # Calculate step distances - from initial separation to overlap
    # Step 0: Initial separation (no contact)
    # Step 1-3: Gradually approach
    # Step 4: Significant overlap
    
    separations = []
    for i in range(num_steps):
        if i == 0:
            # Initial separation - no contact
            separation = initial_separation
        elif i == num_steps - 1:
            # Final step - significant overlap
            separation = -initial_separation * 0.3  # Negative means overlap
        else:
            # Gradual approach
            progress = i / (num_steps - 1)
            separation = initial_separation * (1 - progress * 1.3)  # 1.3 to ensure overlap at the end
        
        separations.append(separation)
    
    print(f"Separation distances: {[f'{s:.4f}' for s in separations]}")
    
    for i, separation in enumerate(separations):
        # Create copies for this step
        cup_step = cup_mesh.copy()
        bowl_step = bowl_mesh.copy()
        
        # Move cup to the target separation distance
        current_cup_center = (cup_step.bounds[0] + cup_step.bounds[1]) / 2
        target_x = separation
        translation = np.array([target_x - current_cup_center[0], 0, 0])
        cup_step.apply_translation(translation)
        
        # Create step description
        if separation > initial_separation * 0.8:
            description = f"Step {i}: Far apart (separation: {separation:.4f})"
        elif separation > 0:
            description = f"Step {i}: Approaching (separation: {separation:.4f})"
        elif separation > -initial_separation * 0.1:
            description = f"Step {i}: Just touching (separation: {separation:.4f})"
        else:
            description = f"Step {i}: Overlapping (penetration: {-separation:.4f})"
        
        steps.append((cup_step, bowl_step, separation, description))
        print(f"  {description}")
    
    return steps

def test_penetration_for_step(step_num, cup_mesh, bowl_mesh, separation, description):
    """
    Test penetration metrics for a single step
    
    Args:
        step_num: Step number
        cup_mesh: Cup mesh for this step
        bowl_mesh: Bowl mesh for this step
        separation: Separation distance
        description: Step description
        
    Returns:
        dict: Penetration metrics for this step
    """
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        # Create mesh list for penetration calculation
        mesh_list = [cup_mesh, bowl_mesh]
        
        # Create merged mesh
        merged_mesh = trimesh.util.concatenate(mesh_list)
        
        print(f"Big-bowl bounds: {cup_mesh.bounds}")
        print(f"Cup-plate bounds: {bowl_mesh.bounds}")
        print(f"Merged bounds: {merged_mesh.bounds}")
        
        # Calculate basic penetration level
        print("\nCalculating basic penetration level...")
        print("Using high-resolution TSDF (80x80x80 = 512K voxels) for better accuracy...")
        try:
            basic_penetration = compute_penetration_level(mesh_list, merged_mesh, size=0.3, resolution=80)
            print(f"✓ Basic penetration level: {basic_penetration:.6f}")
        except Exception as e:
            print(f"✗ Error in basic penetration calculation: {e}")
            basic_penetration = 0.0
        
        # Calculate detailed penetration metrics
        print("\nCalculating detailed penetration metrics...")
        print("Using high-resolution TSDF (80x80x80 = 512K voxels) for detailed analysis...")
        try:
            detailed_metrics = compute_penetration_level_detailed(mesh_list, merged_mesh, size=0.3, resolution=80)
            
            print(f"✓ Detailed penetration metrics:")
            print(f"  - Penetration level: {detailed_metrics['penetration_level']:.6f}")
            print(f"  - Overlap ratio: {detailed_metrics['overlap_ratio']:.6f}")
            print(f"  - Merged internal points: {detailed_metrics['merged_internal_points']}")
            print(f"  - Individual internal points sum: {detailed_metrics['individual_internal_points_sum']}")
            print(f"  - Per-mesh internal points: {detailed_metrics['per_mesh_internal_points']}")
            if 'per_mesh_penetration' in detailed_metrics:
                print(f"  - Per-mesh penetration: {[f'{p:.6f}' for p in detailed_metrics['per_mesh_penetration']]}")
                
        except Exception as e:
            print(f"✗ Error in detailed penetration calculation: {e}")
            detailed_metrics = {
                'penetration_level': 0.0,
                'overlap_ratio': 0.0,
                'merged_internal_points': 0,
                'individual_internal_points_sum': 0,
                'per_mesh_internal_points': [0, 0],
                'per_mesh_penetration': [0.0, 0.0]
            }
        
        # Calculate bounding box IoU
        print("\nCalculating bounding box IoU...")
        try:
            b_iou = compute_bounding_box_iou(cup_mesh, bowl_mesh)
            print(f"✓ Bounding box IoU: {b_iou:.6f}")
        except Exception as e:
            print(f"✗ Error in B-IoU calculation: {e}")
            b_iou = 0.0
        
        # Compile step results
        step_results = {
            'step': step_num,
            'separation': separation,
            'description': description,
            'basic_penetration': basic_penetration,
            'detailed_metrics': detailed_metrics,
            'b_iou': b_iou,
            'cup_bounds': cup_mesh.bounds.tolist(),
            'bowl_bounds': bowl_mesh.bounds.tolist(),
            'merged_bounds': merged_mesh.bounds.tolist()
        }
        
        return step_results
        
    except Exception as e:
        print(f"✗ Error in step {step_num} penetration testing: {e}")
        return {
            'step': step_num,
            'separation': separation,
            'description': description,
            'error': str(e)
        }

def save_step_visualization(step_num, cup_mesh, bowl_mesh, output_dir):
    """
    Save visualization for a step
    
    Args:
        step_num: Step number
        cup_mesh: Cup mesh
        bowl_mesh: Bowl mesh
        output_dir: Output directory
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create scene with both meshes
        scene = trimesh.Scene()
        
        # Color the meshes differently
        cup_colored = cup_mesh.copy()
        bowl_colored = bowl_mesh.copy()
        
        # Set colors: big-bowl = red, cup-plate = blue
        cup_colored.visual.face_colors = [255, 0, 0, 200]  # Red with transparency
        bowl_colored.visual.face_colors = [0, 0, 255, 200]  # Blue with transparency
        
        scene.add_geometry(cup_colored, node_name="big_bowl")
        scene.add_geometry(bowl_colored, node_name="cup_plate")
        
        # Save as GLB file
        output_file = output_dir / f"step_{step_num:02d}_visualization.glb"
        scene.export(str(output_file))
        
        print(f"  ✓ Saved visualization: {output_file}")
        
        # Also save individual meshes
        big_bowl_file = output_dir / f"step_{step_num:02d}_big_bowl.ply"
        cup_plate_file = output_dir / f"step_{step_num:02d}_cup_plate.ply"
        
        cup_mesh.export(str(big_bowl_file))
        bowl_mesh.export(str(cup_plate_file))
        
        print(f"  ✓ Saved individual meshes: {big_bowl_file.name}, {cup_plate_file.name}")
        
        # Save merged/combined mesh
        merged_file = output_dir / f"step_{step_num:02d}_merged.ply"
        merged_mesh = trimesh.util.concatenate([cup_mesh, bowl_mesh])
        merged_mesh.export(str(merged_file))
        
        print(f"  ✓ Saved merged mesh: {merged_file.name}")
        
    except Exception as e:
        print(f"  ✗ Error saving visualization for step {step_num}: {e}")

def save_demo_results(all_results, output_dir):
    """
    Save demo results to files
    
    Args:
        all_results: List of step results
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / "penetration_demo_results.txt"
    
    with open(results_file, 'w') as f:
        f.write("Penetration Demo Results\n")
        f.write("="*50 + "\n\n")
        f.write("Demo: Big-Bowl and Cup-Plate Gradual Approach\n")
        f.write(f"Total steps: {len(all_results)}\n\n")
        
        for result in all_results:
            f.write(f"STEP {result['step']}: {result['description']}\n")
            f.write(f"Separation distance: {result['separation']:.6f}\n")
            
            if 'error' in result:
                f.write(f"ERROR: {result['error']}\n")
            else:
                f.write(f"Basic penetration: {result['basic_penetration']:.6f}\n")
                f.write(f"Detailed penetration level: {result['detailed_metrics']['penetration_level']:.6f}\n")
                f.write(f"Overlap ratio: {result['detailed_metrics']['overlap_ratio']:.6f}\n")
                f.write(f"Bounding box IoU: {result['b_iou']:.6f}\n")
                f.write(f"Merged internal points: {result['detailed_metrics']['merged_internal_points']}\n")
                f.write(f"Individual internal points sum: {result['detailed_metrics']['individual_internal_points_sum']}\n")
                f.write(f"Per-mesh internal points: {result['detailed_metrics']['per_mesh_internal_points']}\n")
            
            f.write("\n" + "-"*40 + "\n\n")
    
    print(f"✓ Detailed results saved to: {results_file}")
    
    # Save summary
    summary_file = output_dir / "penetration_demo_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Penetration Demo Summary\n")
        f.write("="*30 + "\n\n")
        
        f.write("Step | Separation | Basic Pen. | Detailed Pen. | Overlap Ratio | B-IoU\n")
        f.write("-"*75 + "\n")
        
        for result in all_results:
            if 'error' not in result:
                f.write(f"{result['step']:4d} | {result['separation']:10.6f} | {result['basic_penetration']:10.6f} | "
                       f"{result['detailed_metrics']['penetration_level']:13.6f} | {result['detailed_metrics']['overlap_ratio']:13.6f} | "
                       f"{result['b_iou']:5.3f}\n")
            else:
                f.write(f"{result['step']:4d} | {result['separation']:10.6f} | ERROR: {result['error']}\n")
    
    print(f"✓ Summary saved to: {summary_file}")

def main():
    """
    Main function to run the penetration demo
    """
    print("="*60)
    print("PENETRATION SCORE DEMO")
    print("="*60)
    print("Testing penetration between big-bowl.ply and cup-plate.ply")
    print("5 steps from separated to overlapping positions")
    print("="*60)
    
    # Create output directory
    output_dir = Path("messy_kitchen_results/penetration_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # Load demo meshes
        print("\n1. Loading demo meshes...")
        cup_mesh, bowl_mesh = load_demo_meshes()
        
        # Prepare meshes for demo
        print("\n2. Preparing meshes for demo...")
        cup_prepared, bowl_prepared, initial_separation, scale_factor = prepare_meshes_for_demo(cup_mesh, bowl_mesh)
        
        # Create demo steps
        print("\n3. Creating demo steps...")
        demo_steps = create_demo_steps(cup_prepared, bowl_prepared, initial_separation, num_steps=5)
        
        # Test penetration for each step
        print("\n4. Testing penetration for each step...")
        all_results = []
        
        for i, (cup_step, bowl_step, separation, description) in enumerate(demo_steps):
            # Test penetration
            step_result = test_penetration_for_step(i, cup_step, bowl_step, separation, description)
            all_results.append(step_result)
            
            # Save visualization
            save_step_visualization(i, cup_step, bowl_step, output_dir)
        
        # Save results
        print("\n5. Saving demo results...")
        save_demo_results(all_results, output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DEMO SUMMARY")
        print(f"{'='*60}")
        print(f"{'Step':<4} | {'Separation':<10} | {'Basic Pen.':<10} | {'Detailed Pen.':<13} | {'Overlap':<7} | {'B-IoU':<5}")
        print("-"*65)
        
        for result in all_results:
            if 'error' not in result:
                print(f"{result['step']:<4} | {result['separation']:<10.6f} | {result['basic_penetration']:<10.6f} | "
                     f"{result['detailed_metrics']['penetration_level']:<13.6f} | {result['detailed_metrics']['overlap_ratio']:<7.3f} | "
                     f"{result['b_iou']:<5.3f}")
            else:
                print(f"{result['step']:<4} | {result['separation']:<10.6f} | ERROR")
        
        print(f"\n✓ Demo completed successfully!")
        print(f"✓ Results saved to: {output_dir}")
        print(f"✓ Check the visualization files: step_XX_visualization.glb")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
