#!/usr/bin/env python3
"""
Test script for merged mesh alignment functionality
"""

import sys
import os
import numpy as np
import trimesh
from PIL import Image

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.utils.eval_utils import compute_aligned_metrics

def create_test_meshes():
    """Create test meshes for alignment testing"""
    
    # Create a simple cube for GT
    gt_cube1 = trimesh.creation.box(extents=[1, 1, 1])
    gt_cube1.apply_translation([0, 0, 0])
    
    gt_cube2 = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    gt_cube2.apply_translation([2, 0, 0])
    
    # Create GT scene with two objects
    gt_scene = trimesh.Scene([gt_cube1, gt_cube2])
    
    # Create predicted meshes (slightly different)
    pred_cube1 = trimesh.creation.box(extents=[1.1, 1.1, 1.1])
    pred_cube1.apply_translation([0.1, 0.1, 0.1])  # Slight offset
    
    pred_cube2 = trimesh.creation.box(extents=[0.6, 0.6, 0.6])
    pred_cube2.apply_translation([2.1, 0.1, 0.1])  # Slight offset
    
    pred_meshes = [pred_cube1, pred_cube2]
    
    return gt_scene, pred_meshes

def test_merged_alignment():
    """Test the merged mesh alignment functionality"""
    
    print("Creating test meshes...")
    gt_scene, pred_meshes = create_test_meshes()
    
    print(f"GT scene contains {len(gt_scene.dump())} objects")
    print(f"Predicted scene contains {len(pred_meshes)} objects")
    
    print("\nTesting merged mesh alignment...")
    try:
        metrics = compute_aligned_metrics(gt_scene, pred_meshes, num_samples=1000)
        
        print("\nResults:")
        print(f"Chamfer Distance: {metrics['chamfer_distance']:.6f}")
        print(f"F-score: {metrics['f_score']:.6f}")
        print(f"IoU: {metrics['iou']:.6f}")
        print(f"Scene IoU: {metrics['scene_iou']:.6f}")
        
        # Check if merged meshes are available
        if 'aligned_gt_merged' in metrics and 'aligned_pred_merged' in metrics:
            print("\nMerged meshes available:")
            gt_merged = metrics['aligned_gt_merged']
            pred_merged = metrics['aligned_pred_merged']
            print(f"GT merged: {len(gt_merged.vertices)} vertices, {len(gt_merged.faces)} faces")
            print(f"Pred merged: {len(pred_merged.vertices)} vertices, {len(pred_merged.faces)} faces")
            
            # Save test results
            os.makedirs("test_results", exist_ok=True)
            gt_merged.export("test_results/gt_merged_test.glb")
            pred_merged.export("test_results/pred_merged_test.glb")
            print("Test meshes saved to test_results/")
            
        return True
        
    except Exception as e:
        print(f"Error in merged alignment test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_merged_alignment()
    if success:
        print("\n✅ Merged mesh alignment test passed!")
    else:
        print("\n❌ Merged mesh alignment test failed!")


