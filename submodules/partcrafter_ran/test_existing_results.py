#!/usr/bin/env python3
"""
Test script to verify the existing results functionality
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from scripts.eval import PartCrafterEvaluator

def test_existing_results():
    """Test the existing results functionality"""
    
    # Create evaluator (without loading model for testing)
    evaluator = PartCrafterEvaluator(build_gaps=False)
    
    # Test case configuration
    test_config = {
        'mesh_path': 'data/preprocessed_data_messy_kitchen_scenes_test/bb7c492421494988a9abfd8e1accb0cd_combined_fixed.glb',
        'image_path': 'data/preprocessed_data_messy_kitchen_scenes_test/bb7c492421494988a9abfd8e1accb0cd_combined_fixed.png',
        'num_parts': 6
    }
    
    output_dir = 'results/evaluation_messy_kitchen'
    case_name = Path(test_config['mesh_path']).stem
    
    print(f"Testing case: {case_name}")
    print(f"Output directory: {output_dir}")
    
    # Check if existing results exist
    has_existing, mesh_dir, pred_meshes, gt_meshes = evaluator.check_existing_results(case_name, output_dir)
    
    print(f"Has existing results: {has_existing}")
    print(f"Mesh directory: {mesh_dir}")
    print(f"Number of predicted meshes: {len(pred_meshes)}")
    print(f"Number of GT meshes: {len(gt_meshes)}")
    
    if has_existing:
        print("✓ Existing results found and loaded successfully!")
        
        # Check what files exist
        if os.path.exists(mesh_dir):
            files = os.listdir(mesh_dir)
            print(f"Files in mesh directory: {files}")
            
            # Check for key files
            key_files = ['gt_merged.glb', 'pred_merged.glb', 'mesh_summary.json']
            for key_file in key_files:
                file_path = os.path.join(mesh_dir, key_file)
                if os.path.exists(file_path):
                    print(f"✓ Found {key_file}")
                else:
                    print(f"✗ Missing {key_file}")
            
            # Check for individual part files
            pred_part_files = [f for f in files if f.startswith('pred_part_')]
            gt_part_files = [f for f in files if f.startswith('gt_part_')]
            print(f"Found {len(pred_part_files)} predicted part files: {pred_part_files}")
            print(f"Found {len(gt_part_files)} GT part files: {gt_part_files}")
            
            # Check for alignment files
            alignment_files = [f for f in files if 'aligned' in f]
            print(f"Found {len(alignment_files)} alignment files: {alignment_files}")
    else:
        print("✗ No existing results found")
        print("This is expected if the case hasn't been processed yet")

def test_evaluation_workflow():
    """Test the complete evaluation workflow with existing results"""
    
    print("\n" + "="*50)
    print("Testing evaluation workflow with existing results")
    print("="*50)
    
    # Test case configuration
    test_config = {
        'mesh_path': 'data/preprocessed_data_messy_kitchen_scenes_test/bb7c492421494988a9abfd8e1accb0cd_combined_fixed.glb',
        'image_path': 'data/preprocessed_data_messy_kitchen_scenes_test/bb7c492421494988a9abfd8e1accb0cd_combined_fixed.png',
        'num_parts': 6
    }
    
    output_dir = 'results/evaluation_messy_kitchen'
    
    # Create evaluator (without loading model for testing)
    evaluator = PartCrafterEvaluator(build_gaps=False)
    
    print("Testing evaluation with use_existing_results=True, force_inference=False")
    print("This should use existing results and skip inference but still compute metrics")
    
    # Note: We can't actually run the full evaluation without the model loaded
    # But we can test the check_existing_results function
    case_name = Path(test_config['mesh_path']).stem
    has_existing, mesh_dir, pred_meshes, gt_meshes = evaluator.check_existing_results(case_name, output_dir)
    
    if has_existing:
        print(f"✓ Would use existing results for case: {case_name}")
        print(f"  - Found {len(pred_meshes)} predicted meshes")
        print(f"  - Found {len(gt_meshes)} GT meshes")
        print(f"  - Would skip inference")
        print(f"  - Would still compute alignment and metrics")
    else:
        print(f"✗ No existing results found for case: {case_name}")
        print(f"  - Would run inference")
        print(f"  - Would compute alignment and metrics")

if __name__ == "__main__":
    test_existing_results()
    test_evaluation_workflow()
