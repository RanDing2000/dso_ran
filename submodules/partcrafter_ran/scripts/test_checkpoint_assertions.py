#!/usr/bin/env python3
"""
Test script to verify checkpoint assertions in create_partcrafter_dpo_data.py
Test with different checkpoint configurations to ensure assertions work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from create_partcrafter_dpo_data import PartCrafterDPODataCreator

def test_checkpoint_assertions():
    """Test checkpoint assertions in the DPO data creation pipeline"""
    
    print("Testing checkpoint assertions in PartCrafter DPO data creation...")
    
    # Test 1: Valid checkpoint path (should work if exists)
    print("\n=== Test 1: Valid checkpoint path ===")
    try:
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint="/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512",
            checkpoint_iteration=17000
        )
        print("✅ Valid checkpoint path test passed!")
    except AssertionError as e:
        print(f"❌ Valid checkpoint path test failed with assertion: {e}")
    except Exception as e:
        print(f"❌ Valid checkpoint path test failed with error: {e}")
    
    # Test 2: Invalid checkpoint path (should assert)
    print("\n=== Test 2: Invalid checkpoint path ===")
    try:
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint="/nonexistent/path/to/checkpoint",
            checkpoint_iteration=17000
        )
        print("❌ Invalid checkpoint path test failed - should have asserted!")
    except AssertionError as e:
        print(f"✅ Invalid checkpoint path test passed with expected assertion: {e}")
    except Exception as e:
        print(f"❌ Invalid checkpoint path test failed with unexpected error: {e}")
    
    # Test 3: None checkpoint path (should work with pretrained model)
    print("\n=== Test 3: None checkpoint path ===")
    try:
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint=None,
            checkpoint_iteration=None
        )
        print("✅ None checkpoint path test passed!")
    except Exception as e:
        print(f"❌ None checkpoint path test failed: {e}")
    
    # Test 4: Partial checkpoint parameters (should assert)
    print("\n=== Test 4: Partial checkpoint parameters ===")
    try:
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint="/some/path",
            checkpoint_iteration=None
        )
        print("❌ Partial checkpoint parameters test failed - should have asserted!")
    except AssertionError as e:
        print(f"✅ Partial checkpoint parameters test passed with expected assertion: {e}")
    except Exception as e:
        print(f"❌ Partial checkpoint parameters test failed with unexpected error: {e}")
    
    print("\n🎉 Checkpoint assertions test completed!")

if __name__ == "__main__":
    test_checkpoint_assertions()
