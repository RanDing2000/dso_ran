#!/usr/bin/env python3
"""
Test script to verify the alignment direction (pred to gt)
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.utils.eval_utils import align_merged_meshes_with_gaps, compute_aligned_metrics
import trimesh
import numpy as np

def test_alignment_direction():
    """Test that predicted mesh is aligned to GT mesh (not the other way around)"""
    
    print("="*60)
    print("Testing Alignment Direction: Pred to GT")
    print("="*60)
    
    # Create test meshes with known transformations
    print("Creating test meshes with known transformations...")
    
    # Create GT mesh (reference)
    gt_mesh = trimesh.creation.box(extents=[2, 2, 2])
    gt_mesh.vertices += [0, 0, 0]  # Center at origin
    
    # Create predicted mesh with known offset
    pred_mesh = trimesh.creation.box(extents=[2, 2, 2])
    pred_mesh.vertices += [1, 1, 1]  # Offset by [1, 1, 1]
    
    print(f"✓ Created test meshes")
    print(f"  - GT mesh center: {np.mean(gt_mesh.vertices, axis=0)}")
    print(f"  - Pred mesh center: {np.mean(pred_mesh.vertices, axis=0)}")
    print(f"  - Expected alignment: pred should move to match gt")
    
    # Test alignment
    try:
        aligned_gt, aligned_pred = align_merged_meshes_with_gaps(gt_mesh, pred_mesh)
        
        print(f"✓ Alignment completed")
        print(f"  - GT mesh center (should be unchanged): {np.mean(aligned_gt.vertices, axis=0)}")
        print(f"  - Aligned pred mesh center: {np.mean(aligned_pred.vertices, axis=0)}")
        
        # Check if alignment worked
        gt_center = np.mean(aligned_gt.vertices, axis=0)
        pred_center = np.mean(aligned_pred.vertices, axis=0)
        
        distance = np.linalg.norm(gt_center - pred_center)
        print(f"  - Distance between centers: {distance:.6f}")
        
        if distance < 0.1:  # Should be very close after alignment
            print(f"  ✓ Alignment successful: predicted mesh aligned to GT mesh")
        else:
            print(f"  ✗ Alignment may have failed: centers still far apart")
            
    except Exception as e:
        print(f"✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()

def test_complete_flow_direction():
    """Test the complete flow to ensure correct alignment direction"""
    
    print("\n" + "="*60)
    print("Testing Complete Flow Alignment Direction")
    print("="*60)
    
    # Create multiple test meshes
    print("Creating multiple test meshes...")
    
    # GT meshes (reference)
    gt_mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    gt_mesh1.vertices += [0, 0, 0]
    gt_mesh2 = trimesh.creation.sphere(radius=0.5)
    gt_mesh2.vertices += [2, 0, 0]
    
    # Predicted meshes (with offset)
    pred_mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    pred_mesh1.vertices += [0.5, 0.5, 0.5]  # Offset
    pred_mesh2 = trimesh.creation.sphere(radius=0.5)
    pred_mesh2.vertices += [2.5, 0.5, 0.5]  # Offset
    
    gt_meshes = [gt_mesh1, gt_mesh2]
    pred_meshes = [pred_mesh1, pred_mesh2]
    
    print(f"✓ Created test meshes")
    print(f"  - GT meshes: {len(gt_meshes)} meshes")
    print(f"  - Pred meshes: {len(pred_meshes)} meshes")
    
    # Test complete flow
    try:
        metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=1000)
        
        print(f"✓ Complete flow successful")
        print(f"  - Scene CD: {metrics['chamfer_distance']:.6f}")
        print(f"  - Scene F-score: {metrics['f_score']:.6f}")
        print(f"  - Scene IoU: {metrics['iou']:.6f}")
        
        # Check if aligned scenes are available
        if 'aligned_gt_scene' in metrics and 'aligned_pred_scene' in metrics:
            print(f"✓ Aligned scenes available")
            
            # Check centers of aligned scenes
            gt_scene = metrics['aligned_gt_scene']
            pred_scene = metrics['aligned_pred_scene']
            
            gt_meshes_aligned = list(gt_scene.geometry.values())
            pred_meshes_aligned = list(pred_scene.geometry.values())
            
            print(f"  - GT meshes in aligned scene: {len(gt_meshes_aligned)}")
            print(f"  - Pred meshes in aligned scene: {len(pred_meshes_aligned)}")
            
            # Check if GT meshes are unchanged (reference)
            for i, (gt_orig, gt_aligned) in enumerate(zip(gt_meshes, gt_meshes_aligned)):
                orig_center = np.mean(gt_orig.vertices, axis=0)
                aligned_center = np.mean(gt_aligned.vertices, axis=0)
                distance = np.linalg.norm(orig_center - aligned_center)
                print(f"    GT mesh {i}: distance from original = {distance:.6f} (should be ~0)")
                
            # Check if pred meshes moved (aligned)
            for i, (pred_orig, pred_aligned) in enumerate(zip(pred_meshes, pred_meshes_aligned)):
                orig_center = np.mean(pred_orig.vertices, axis=0)
                aligned_center = np.mean(pred_aligned.vertices, axis=0)
                distance = np.linalg.norm(orig_center - aligned_center)
                print(f"    Pred mesh {i}: distance from original = {distance:.6f} (should be >0)")
        
    except Exception as e:
        print(f"✗ Complete flow failed: {e}")
        import traceback
        traceback.print_exc()

def test_with_existing_data():
    """Test with existing data to verify alignment direction"""
    
    print("\n" + "="*60)
    print("Testing with Existing Data")
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
        
        # Test alignment direction
        print("Testing alignment direction with real data...")
        
        # Store original centers
        gt_orig_centers = [np.mean(mesh.vertices, axis=0) for mesh in gt_meshes]
        pred_orig_centers = [np.mean(mesh.vertices, axis=0) for mesh in pred_meshes]
        
        # Run alignment
        metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=1000)
        
        print(f"✓ Alignment completed")
        
        # Check alignment results
        if 'aligned_gt_scene' in metrics and 'aligned_pred_scene' in metrics:
            gt_scene = metrics['aligned_gt_scene']
            pred_scene = metrics['aligned_pred_scene']
            
            gt_meshes_aligned = list(gt_scene.geometry.values())
            pred_meshes_aligned = list(pred_scene.geometry.values())
            
            print(f"  - GT meshes: {len(gt_meshes_aligned)}")
            print(f"  - Pred meshes: {len(pred_meshes_aligned)}")
            
            # Check GT meshes (should be unchanged)
            for i, (orig_center, mesh) in enumerate(zip(gt_orig_centers, gt_meshes_aligned)):
                aligned_center = np.mean(mesh.vertices, axis=0)
                distance = np.linalg.norm(orig_center - aligned_center)
                print(f"    GT mesh {i}: distance = {distance:.6f} (should be ~0)")
            
            # Check pred meshes (should be aligned)
            for i, (orig_center, mesh) in enumerate(zip(pred_orig_centers, pred_meshes_aligned)):
                aligned_center = np.mean(mesh.vertices, axis=0)
                distance = np.linalg.norm(orig_center - aligned_center)
                print(f"    Pred mesh {i}: distance = {distance:.6f} (should be >0)")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alignment_direction()
    test_complete_flow_direction()
    test_with_existing_data()
