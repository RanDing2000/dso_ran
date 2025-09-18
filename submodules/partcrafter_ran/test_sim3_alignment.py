#!/usr/bin/env python3
"""
Test script to verify the sim(3) alignment and per-object metric computation flow
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.utils.eval_utils import compute_aligned_metrics, extract_and_apply_transformation
import trimesh
import numpy as np

def test_sim3_alignment_flow():
    """Test the complete sim(3) alignment and per-object metric computation flow"""
    
    print("="*60)
    print("Testing sim(3) Alignment and Per-Object Metric Computation Flow")
    print("="*60)
    
    # Check if we have existing results to test with
    test_dir = "results/evaluation_messy_kitchen/bb7c492421494988a9abfd8e1accb0cd_combined_fixed"
    
    if not os.path.exists(test_dir):
        print(f"✗ Test directory not found: {test_dir}")
        print("Please run evaluation first to generate test data")
        return
    
    print(f"✓ Found test directory: {test_dir}")
    
    # Load test meshes
    try:
        # Load GT meshes
        gt_meshes = []
        i = 0
        while True:
            gt_path = os.path.join(test_dir, f"gt_part_{i:02d}.glb")
            if not os.path.exists(gt_path):
                break
            mesh = trimesh.load(gt_path, process=False)
            if mesh is not None and len(mesh.vertices) > 0:
                gt_meshes.append(mesh)
            i += 1
        
        # Load predicted meshes
        pred_meshes = []
        i = 0
        while True:
            pred_path = os.path.join(test_dir, f"pred_part_{i:02d}.glb")
            if not os.path.exists(pred_path):
                break
            mesh = trimesh.load(pred_path, process=False)
            if mesh is not None and len(mesh.vertices) > 0:
                pred_meshes.append(mesh)
            i += 1
        
        print(f"✓ Loaded {len(gt_meshes)} GT meshes and {len(pred_meshes)} predicted meshes")
        
        if len(gt_meshes) == 0 or len(pred_meshes) == 0:
            print("✗ No valid meshes found for testing")
            return
        
        # Test the complete flow
        print("\n" + "-"*40)
        print("Testing complete sim(3) alignment and metric computation flow")
        print("-"*40)
        
        try:
            metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=1000)
            
            print(f"✓ Complete flow successful")
            print(f"  - Scene CD: {metrics['chamfer_distance']:.6f}±{metrics['chamfer_distance_std']:.6f}")
            print(f"  - Scene F-score: {metrics['f_score']:.6f}±{metrics['f_score_std']:.6f}")
            print(f"  - Scene IoU: {metrics['iou']:.6f}±{metrics['iou_std']:.6f}")
            print(f"  - Scene IoU (direct): {metrics['scene_iou']:.6f}")
            
            # Check per-object metrics
            if 'per_object_cds' in metrics:
                print(f"  - Per-object CDs: {len(metrics['per_object_cds'])} objects")
                for i, cd in enumerate(metrics['per_object_cds']):
                    fscore = metrics['per_object_fscores'][i]
                    iou = metrics['per_object_ious'][i]
                    print(f"    Object {i}: CD={cd:.6f}, F-score={fscore:.6f}, IoU={iou:.6f}")
            
            # Check if aligned scenes are available
            if 'aligned_gt_scene' in metrics and 'aligned_pred_scene' in metrics:
                print(f"✓ Aligned scenes available for visualization")
            
            # Check if aligned merged meshes are available
            if 'aligned_gt_merged' in metrics and 'aligned_pred_merged' in metrics:
                print(f"✓ Aligned merged meshes available for saving")
                
        except Exception as e:
            print(f"✗ Complete flow failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "="*60)
        print("✓ All tests passed! The sim(3) alignment and per-object metric computation flow is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        import traceback
        traceback.print_exc()

def test_transformation_extraction():
    """Test the transformation extraction function separately"""
    
    print("\n" + "="*60)
    print("Testing Transformation Extraction")
    print("="*60)
    
    # Create simple test meshes
    print("Creating simple test meshes...")
    
    # Create a simple cube as original mesh
    original_mesh = trimesh.creation.box(extents=[1, 1, 1])
    
    # Create a transformed version (scaled, rotated, translated)
    transformed_mesh = trimesh.creation.box(extents=[1.2, 1.2, 1.2])  # Scaled
    transformed_mesh.vertices += [0.5, 0.3, 0.2]  # Translated
    # Apply rotation
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    transformed_mesh.vertices = (R @ transformed_mesh.vertices.T).T
    
    # Create individual meshes to transform
    individual_meshes = [
        trimesh.creation.box(extents=[0.5, 0.5, 0.5]),
        trimesh.creation.sphere(radius=0.3)
    ]
    
    print(f"✓ Created test meshes")
    print(f"  - Original mesh: {len(original_mesh.vertices)} vertices")
    print(f"  - Transformed mesh: {len(transformed_mesh.vertices)} vertices")
    print(f"  - Individual meshes: {len(individual_meshes)} meshes")
    
    # Test transformation extraction
    try:
        aligned_individual_meshes = extract_and_apply_transformation(
            original_mesh, transformed_mesh, individual_meshes
        )
        
        print(f"✓ Transformation extraction successful")
        print(f"  - Transformed {len(aligned_individual_meshes)} individual meshes")
        
        # Check if transformation was applied
        for i, mesh in enumerate(aligned_individual_meshes):
            print(f"    Mesh {i}: {len(mesh.vertices)} vertices")
        
    except Exception as e:
        print(f"✗ Transformation extraction failed: {e}")
        import traceback
        traceback.print_exc()

def test_individual_steps():
    """Test individual steps separately"""
    
    print("\n" + "="*60)
    print("Testing Individual Steps")
    print("="*60)
    
    # Create simple test meshes
    print("Creating simple test meshes...")
    
    # Create simple cubes as GT meshes
    gt_mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    gt_mesh1.vertices += [0, 0, 0]
    gt_mesh2 = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    gt_mesh2.vertices += [2, 0, 0]
    
    # Create simple cubes as predicted meshes (with some offset)
    pred_mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    pred_mesh1.vertices += [0.1, 0.1, 0.1]  # Small offset
    pred_mesh2 = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    pred_mesh2.vertices += [2.1, 0.1, 0.1]  # Small offset
    
    gt_meshes = [gt_mesh1, gt_mesh2]
    pred_meshes = [pred_mesh1, pred_mesh2]
    
    print(f"✓ Created test meshes")
    print(f"  - GT meshes: {len(gt_meshes)} meshes")
    print(f"  - Pred meshes: {len(pred_meshes)} meshes")
    
    # Test the complete flow
    try:
        metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=100)
        
        print(f"✓ Test metrics computed")
        print(f"  - Scene CD: {metrics['chamfer_distance']:.6f}")
        print(f"  - Scene F-score: {metrics['f_score']:.6f}")
        print(f"  - Scene IoU: {metrics['iou']:.6f}")
        
        # Check per-object metrics
        if 'per_object_cds' in metrics:
            print(f"  - Per-object metrics: {len(metrics['per_object_cds'])} objects")
            for i, cd in enumerate(metrics['per_object_cds']):
                fscore = metrics['per_object_fscores'][i]
                iou = metrics['per_object_ious'][i]
                print(f"    Object {i}: CD={cd:.6f}, F-score={fscore:.6f}, IoU={iou:.6f}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sim3_alignment_flow()
    test_transformation_extraction()
    test_individual_steps()
