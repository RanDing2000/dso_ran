#!/usr/bin/env python3
"""
Test script to verify checkpoint loading in create_partcrafter_dpo_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from create_partcrafter_dpo_data import PartCrafterDPODataCreator

def test_checkpoint_loading():
    """Test checkpoint loading with default parameters"""
    
    print("Testing checkpoint loading in PartCrafter DPO data creation...")
    
    try:
        # Test with default checkpoint parameters (as set in the script)
        print("Creating DPO data creator with default checkpoint parameters...")
        
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint="/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512",
            checkpoint_iteration=17000
        )
        
        print("✅ Checkpoint loading test passed!")
        print("✅ DPO data creator initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_checkpoint_loading()
    if success:
        print("\n🎉 Checkpoint loading test completed successfully!")
    else:
        print("\n💥 Checkpoint loading test failed!")
        sys.exit(1)
