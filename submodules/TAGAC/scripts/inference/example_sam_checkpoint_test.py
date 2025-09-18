#!/usr/bin/env python3
"""
Example script showing how to test multiple SAM checkpoints with TARGO.
This script demonstrates the enhanced robustness features.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/projects/TARGO')

def test_single_sam_checkpoint():
    """Test a single SAM checkpoint."""
    print("=" * 60)
    print("TESTING SINGLE SAM CHECKPOINT")
    print("=" * 60)
    
    cmd = [
        "python", "scripts/inference/inference_vgn_targo_sam.py",
        "--model_type", "targo",
        "--occlusion-level", "slight",
        "--max_scenes", "5",
        "--vis", "true",
        "--sam_checkpoint", "checkpoints/sam_vit_h_4b8939.pth"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out after 5 minutes")
    except Exception as e:
        print(f"Error running command: {e}")


def test_multiple_sam_checkpoints():
    """Test multiple SAM checkpoints from a directory."""
    print("=" * 60)
    print("TESTING MULTIPLE SAM CHECKPOINTS")
    print("=" * 60)
    
    # Create a test directory with multiple checkpoints (if they exist)
    sam_checkpoint_dir = "checkpoints/sam_test"
    
    if not os.path.exists(sam_checkpoint_dir):
        print(f"Creating test directory: {sam_checkpoint_dir}")
        os.makedirs(sam_checkpoint_dir, exist_ok=True)
        
        # Create dummy checkpoint files for testing
        dummy_checkpoints = [
            "sam_vit_h_test.pth",
            "sam_vit_l_test.pth", 
            "sam_vit_b_test.pth"
        ]
        
        for checkpoint in dummy_checkpoints:
            checkpoint_path = os.path.join(sam_checkpoint_dir, checkpoint)
            with open(checkpoint_path, 'w') as f:
                f.write("# Dummy checkpoint file for testing\n")
            print(f"Created dummy checkpoint: {checkpoint_path}")
    
    cmd = [
        "python", "scripts/inference/inference_vgn_targo_sam.py",
        "--model_type", "targo",
        "--occlusion-level", "slight", 
        "--max_scenes", "3",
        "--vis", "false",  # Disable visualization for faster testing
        "--sam_checkpoint_dir", sam_checkpoint_dir
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out after 10 minutes")
    except Exception as e:
        print(f"Error running command: {e}")


def create_test_script():
    """Create the test script for multiple SAM checkpoints."""
    print("=" * 60)
    print("CREATING SAM CHECKPOINT TEST SCRIPT")
    print("=" * 60)
    
    cmd = [
        "python", "scripts/inference/inference_vgn_targo_sam.py",
        "--create-test-script"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running command: {e}")


def main():
    """Main function to run all tests."""
    print("SAM CHECKPOINT TESTING EXAMPLES")
    print("=" * 60)
    
    # Test 1: Create test script
    create_test_script()
    print("\n")
    
    # Test 2: Single checkpoint (commented out to avoid actual model loading)
    # test_single_sam_checkpoint()
    # print("\n")
    
    # Test 3: Multiple checkpoints (commented out to avoid actual model loading)
    # test_multiple_sam_checkpoints()
    
    print("=" * 60)
    print("USAGE EXAMPLES:")
    print("=" * 60)
    print("1. Test single SAM checkpoint:")
    print("   python scripts/inference/inference_vgn_targo_sam.py \\")
    print("       --model_type targo \\")
    print("       --occlusion-level slight \\")
    print("       --max_scenes 10 \\")
    print("       --vis true \\")
    print("       --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth")
    print()
    print("2. Test multiple SAM checkpoints:")
    print("   python scripts/inference/inference_vgn_targo_sam.py \\")
    print("       --model_type targo \\")
    print("       --occlusion-level slight \\")
    print("       --max_scenes 10 \\")
    print("       --vis true \\")
    print("       --sam_checkpoint_dir checkpoints/sam_checkpoints/")
    print()
    print("3. Create test script:")
    print("   python scripts/inference/inference_vgn_targo_sam.py --create-test-script")
    print()
    print("4. Run test script:")
    print("   ./test_sam_checkpoints.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()

