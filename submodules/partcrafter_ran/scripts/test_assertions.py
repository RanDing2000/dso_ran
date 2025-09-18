#!/usr/bin/env python3
"""
Test script to verify assertions in create_partcrafter_dpo_data.py
Test with minimal parameters to ensure assertions work correctly
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from create_partcrafter_dpo_data import PartCrafterDPODataCreator

def test_assertions():
    """Test assertions in the DPO data creation pipeline"""
    
    print("Testing assertions in PartCrafter DPO data creation...")
    
    # Find a test scene
    test_scene_dir = Path("/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/messy_kitchen_scenes_renderings")
    scene_dirs = [d for d in test_scene_dir.iterdir() if d.is_dir()]
    
    if not scene_dirs:
        print("No test scenes found!")
        return
    
    test_scene = scene_dirs[0]
    print(f"Using test scene: {test_scene.name}")
    
    try:
        # Create DPO data creator (using pretrained model)
        print("Initializing DPO data creator...")
        
        creator = PartCrafterDPODataCreator(
            model_path="pretrained_weights/PartCrafter",
            device="cuda",
            dtype=float,
            load_trained_checkpoint=None,
            checkpoint_iteration=None
        )
        
        print("Testing run_inference with assertions...")
        
        # Test run_inference with minimal parameters to trigger assertions if needed
        image_path = str(test_scene / "rendering.png")
        num_parts = 2  # Use a small number for testing
        
        try:
            pred_meshes, input_image, cond_features, latent_features = creator.run_inference(
                image_path=image_path,
                num_parts=num_parts,
                seed=0,
                num_tokens=256,  # Reduced for faster testing
                num_inference_steps=5,  # Very reduced for faster testing
                guidance_scale=7.0,
                rmbg=False,
            )
            
            print("✅ run_inference completed successfully!")
            print(f"Number of meshes: {len(pred_meshes)}")
            print(f"Cond features shape: {cond_features.shape}")
            print(f"Latent features shape: {latent_features.shape}")
            
            # Verify that we have valid outputs
            assert len(pred_meshes) == num_parts, f"Expected {num_parts} meshes, got {len(pred_meshes)}"
            assert cond_features is not None, "Cond features should not be None"
            assert latent_features is not None, "Latent features should not be None"
            assert latent_features.shape[0] == num_parts, f"Expected {num_parts} parts in latent features, got {latent_features.shape[0]}"
            
            print("✅ All assertions passed!")
            
        except AssertionError as e:
            print(f"❌ Assertion failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Other error occurred: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("Feature extraction test with assertions completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_assertions()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)
