#!/usr/bin/env python3
"""
Test script to verify the GAPS alignment and metric computation flow
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.utils.eval_utils import compute_aligned_metrics, align_meshes_with_gaps
import trimesh

def test_alignment_flow():
    """Test the complete alignment and metric computation flow"""
    
    print("="*60)
    print("Testing GAPS Alignment and Metric Computation Flow")
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
        
        # Test Step 1: GAPS Alignment
        print("\n" + "-"*40)
        print("Step 1: Testing GAPS Alignment")
        print("-"*40)
        
        try:
            aligned_gt_scene, aligned_pred_scene = align_meshes_with_gaps(gt_meshes, pred_meshes)
            print(f"✓ GAPS alignment successful")
            print(f"  - Aligned GT scene: {len(aligned_gt_scene.geometry)} objects")
            print(f"  - Aligned pred scene: {len(aligned_pred_scene.geometry)} objects")
        except Exception as e:
            print(f"✗ GAPS alignment failed: {e}")
            return
        
        # Test Step 2: Merging aligned meshes
        print("\n" + "-"*40)
        print("Step 2: Testing Mesh Merging")
        print("-"*40)
        
        aligned_gt_meshes = list(aligned_gt_scene.geometry.values())
        aligned_pred_meshes = list(aligned_pred_scene.geometry.values())
        
        # Filter out empty meshes
        aligned_gt_meshes = [mesh for mesh in aligned_gt_meshes if mesh is not None and len(mesh.vertices) > 0]
        aligned_pred_meshes = [mesh for mesh in aligned_pred_meshes if mesh is not None and len(mesh.vertices) > 0]
        
        print(f"✓ Filtered aligned meshes")
        print(f"  - GT meshes: {len(aligned_gt_meshes)}")
        print(f"  - Pred meshes: {len(aligned_pred_meshes)}")
        
        # Merge aligned meshes
        if len(aligned_gt_meshes) > 1:
            aligned_gt_merged = trimesh.util.concatenate(aligned_gt_meshes)
        else:
            aligned_gt_merged = aligned_gt_meshes[0] if aligned_gt_meshes else trimesh.Trimesh()
        
        if len(aligned_pred_meshes) > 1:
            aligned_pred_merged = trimesh.util.concatenate(aligned_pred_meshes)
        else:
            aligned_pred_merged = aligned_pred_meshes[0] if aligned_pred_meshes else trimesh.Trimesh()
        
        print(f"✓ Merged aligned meshes")
        print(f"  - GT merged: {len(aligned_gt_merged.vertices)} vertices, {len(aligned_gt_merged.faces)} faces")
        print(f"  - Pred merged: {len(aligned_pred_merged.vertices)} vertices, {len(aligned_pred_merged.faces)} faces")
        
        # Test Step 3: Metric computation
        print("\n" + "-"*40)
        print("Step 3: Testing Metric Computation")
        print("-"*40)
        
        try:
            # Test the complete flow
            metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=1000)
            
            print(f"✓ Metric computation successful")
            print(f"  - Chamfer Distance: {metrics['chamfer_distance']:.6f}")
            print(f"  - F-score: {metrics['f_score']:.6f}")
            print(f"  - IoU: {metrics['iou']:.6f}")
            print(f"  - Scene IoU: {metrics['scene_iou']:.6f}")
            
            # Check if aligned scenes are available
            if 'aligned_gt_scene' in metrics and 'aligned_pred_scene' in metrics:
                print(f"✓ Aligned scenes available for visualization")
            
            # Check if aligned merged meshes are available
            if 'aligned_gt_merged' in metrics and 'aligned_pred_merged' in metrics:
                print(f"✓ Aligned merged meshes available for saving")
                
        except Exception as e:
            print(f"✗ Metric computation failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "="*60)
        print("✓ All tests passed! The alignment and metric computation flow is working correctly.")
        print("="*60)
        
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        import traceback
        traceback.print_exc()

def test_individual_steps():
    """Test individual steps separately"""
    
    print("\n" + "="*60)
    print("Testing Individual Steps")
    print("="*60)
    
    # Create simple test meshes
    print("Creating simple test meshes...")
    
    # Create a simple cube as GT mesh
    gt_mesh = trimesh.creation.box(extents=[1, 1, 1])
    gt_mesh.vertices += [0.1, 0.1, 0.1]  # Small offset
    
    # Create a simple cube as predicted mesh
    pred_mesh = trimesh.creation.box(extents=[1, 1, 1])
    pred_mesh.vertices += [0.2, 0.2, 0.2]  # Different offset
    
    gt_meshes = [gt_mesh]
    pred_meshes = [pred_mesh]
    
    print(f"✓ Created test meshes")
    print(f"  - GT mesh: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
    print(f"  - Pred mesh: {len(pred_mesh.vertices)} vertices, {len(pred_mesh.faces)} faces")
    
    # Test the complete flow
    try:
        metrics = compute_aligned_metrics(gt_meshes, pred_meshes, num_samples=100)
        
        print(f"✓ Test metrics computed")
        print(f"  - Chamfer Distance: {metrics['chamfer_distance']:.6f}")
        print(f"  - F-score: {metrics['f_score']:.6f}")
        print(f"  - IoU: {metrics['iou']:.6f}")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alignment_flow()
    test_individual_steps()
