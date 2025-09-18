#!/usr/bin/env python3
"""
Test script for PartCrafter feature extraction
Test cond and latent feature extraction with minimal setup
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from create_partcrafter_dpo_data import PartCrafterDPODataCreator

def test_feature_extraction():
    """Test cond and latent feature extraction"""
    
    print("Testing PartCrafter feature extraction...")
    
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
        
        print("Testing feature extraction...")
        
        # Test run_inference with feature extraction
        image_path = str(test_scene / "rendering.png")
        num_parts = 3  # Use a small number for testing
        
        pred_meshes, input_image, cond_features, latent_features = creator.run_inference(
            image_path=image_path,
            num_parts=num_parts,
            seed=0,
            num_tokens=512,  # Reduced for faster testing
            num_inference_steps=10,  # Reduced for faster testing
            guidance_scale=7.0,
            rmbg=False,
        )
        
        print("Feature extraction completed!")
        print(f"Number of meshes: {len(pred_meshes)}")
        print(f"Cond features shape: {cond_features.shape}")
        print(f"Latent features shape: {latent_features.shape}")
        print(f"Cond features dtype: {cond_features.dtype}")
        print(f"Latent features dtype: {latent_features.dtype}")
        
        # Test saving features
        output_dir = Path("/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/test_features")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cond features
        cond_path = output_dir / "test_cond.pt"
        import torch
        torch.save(cond_features.cpu().to(torch.bfloat16), cond_path)
        print(f"Saved cond features to: {cond_path}")
        
        # Save latent features
        for i in range(num_parts):
            latent_path = output_dir / f"test_latent_{i:03d}.pt"
            if i < latent_features.shape[0]:
                part_latent = latent_features[i].cpu().to(torch.bfloat16)
                torch.save(part_latent, latent_path)
                print(f"Saved latent features part {i} to: {latent_path}")
        
        print("Feature extraction test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_extraction()
