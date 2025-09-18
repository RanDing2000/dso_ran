#!/usr/bin/env python3
"""
Test script to verify mesh generation in create_partcrafter_dpo_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from create_partcrafter_dpo_data import PartCrafterDPODataCreator

def test_mesh_generation():
    """Test mesh generation with minimal parameters"""
    
    print("Testing mesh generation in PartCrafter DPO data creation...")
    
    # Find a test scene
    test_scene_dir = Path("/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/messy_kitchen_scenes_renderings")
    scene_dirs = [d for d in test_scene_dir.iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print("No test scenes found!")
        return False
    
    test_scene = scene_dirs[0]
    print(f"Using test scene: {test_scene.name}")
    
    try:
        # Create DPO data creator
        print("Creating DPO data creator...")
        
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint="/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512",
            checkpoint_iteration=17000
        )
        
        print("Testing mesh generation...")
        
        # Test run_inference with minimal parameters
        image_path = str(test_scene / "rendering.png")
        num_parts = 2  # Use a small number for testing
        
        pred_meshes, input_image, cond_features, latent_features = creator.run_inference(
            image_path=image_path,
            num_parts=num_parts,
            seed=0,
            num_tokens=256,  # Reduced for faster testing
            num_inference_steps=10,  # Reduced for faster testing
            guidance_scale=7.0,
            rmbg=False,
        )
        
        print("✅ Mesh generation test completed!")
        print(f"Number of meshes: {len(pred_meshes)}")
        print(f"Cond features shape: {cond_features.shape}")
        print(f"Latent features shape: {latent_features.shape}")
        
        # Check that we have valid outputs
        assert len(pred_meshes) == num_parts, f"Expected {num_parts} meshes, got {len(pred_meshes)}"
        assert cond_features is not None, "Cond features should not be None"
        assert latent_features is not None, "Latent features should not be None"
        
        print("✅ All outputs are valid!")
        
        return True
        
    except Exception as e:
        print(f"❌ Mesh generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mesh_generation()
    if success:
        print("\n🎉 Mesh generation test completed successfully!")
    else:
        print("\n💥 Mesh generation test failed!")
        sys.exit(1)
