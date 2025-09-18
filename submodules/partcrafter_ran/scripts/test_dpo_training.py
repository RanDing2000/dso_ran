#!/usr/bin/env python3
"""
Test script for PartCrafter DPO Training
This script tests the basic functionality of the DPO training pipeline
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Add PartCrafter to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

def test_preference_pairs_loading():
    """Test if preference pairs can be loaded correctly"""
    preference_pairs_file = "/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json"
    
    if not os.path.exists(preference_pairs_file):
        print(f"❌ Preference pairs file not found: {preference_pairs_file}")
        return False
    
    try:
        with open(preference_pairs_file, 'r') as f:
            data = json.load(f)
        
        preference_pairs = data.get('preference_pairs', [])
        print(f"✅ Successfully loaded {len(preference_pairs)} preference pairs")
        
        if len(preference_pairs) > 0:
            sample_pair = preference_pairs[0]
            required_keys = ['win_seed_dir', 'loss_seed_dir', 'penetration_diff', 'scene_name']
            
            for key in required_keys:
                if key not in sample_pair:
                    print(f"❌ Missing required key '{key}' in preference pair")
                    return False
            
            print(f"✅ Sample preference pair structure is valid")
            print(f"   Scene: {sample_pair['scene_name']}")
            print(f"   Penetration diff: {sample_pair['penetration_diff']:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading preference pairs: {e}")
        return False

def test_data_loading():
    """Test if DPO data files exist and can be loaded"""
    preference_pairs_file = "/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json"
    
    try:
        with open(preference_pairs_file, 'r') as f:
            data = json.load(f)
        
        preference_pairs = data.get('preference_pairs', [])
        
        if len(preference_pairs) == 0:
            print("❌ No preference pairs found")
            return False
        
        # Test loading a few sample data files
        test_count = min(3, len(preference_pairs))
        success_count = 0
        
        for i in range(test_count):
            pair = preference_pairs[i]
            win_seed_dir = pair['win_seed_dir']
            loss_seed_dir = pair['loss_seed_dir']
            
            # Check if directories exist
            if not os.path.exists(win_seed_dir):
                print(f"❌ Win seed directory not found: {win_seed_dir}")
                continue
                
            if not os.path.exists(loss_seed_dir):
                print(f"❌ Loss seed directory not found: {loss_seed_dir}")
                continue
            
            # Check for required files
            cond_path = os.path.join(win_seed_dir, 'cond.pt')
            if not os.path.exists(cond_path):
                print(f"❌ Cond features not found: {cond_path}")
                continue
            
            # Check for latent features
            latent_files = [f for f in os.listdir(win_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
            if len(latent_files) == 0:
                print(f"❌ No latent features found in: {win_seed_dir}")
                continue
            
            try:
                # Try loading the files
                cond_features = torch.load(cond_path, map_location='cpu')
                print(f"✅ Loaded cond features: {cond_features.shape}")
                
                # Load first latent file
                latent_path = os.path.join(win_seed_dir, sorted(latent_files)[0])
                latent_features = torch.load(latent_path, map_location='cpu')
                print(f"✅ Loaded latent features: {latent_features.shape}")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error loading files for pair {i}: {e}")
                continue
        
        print(f"✅ Successfully tested {success_count}/{test_count} data samples")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ Error in data loading test: {e}")
        return False

def test_model_imports():
    """Test if all required model imports work"""
    try:
        from src.models.transformers import PartCrafterDiTModel
        from src.models.autoencoders import TripoSGVAEModel
        from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
        from transformers import Dinov2Model, BitImageProcessor
        print("✅ All model imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test if models can be loaded from checkpoint"""
    try:
        from src.models.transformers import PartCrafterDiTModel
        from src.models.autoencoders import TripoSGVAEModel
        from transformers import Dinov2Model, BitImageProcessor
        
        checkpoint_path = "/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512/checkpoints/017000"
        
        print("🔄 Testing model loading...")
        
        # Test transformer loading
        transformer = PartCrafterDiTModel.from_pretrained(checkpoint_path, subfolder="transformer")
        print("✅ Transformer loaded successfully")
        
        # Test VAE loading
        vae = TripoSGVAEModel.from_pretrained(checkpoint_path, subfolder="vae")
        print("✅ VAE loaded successfully")
        
        # Test image encoder loading
        image_encoder = Dinov2Model.from_pretrained(checkpoint_path, subfolder="image_encoder_dinov2")
        print("✅ Image encoder loaded successfully")
        
        # Test image processor loading
        image_processor = BitImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor_dinov2")
        print("✅ Image processor loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_checkpoint_path():
    """Test if checkpoint path exists"""
    checkpoint_path = "/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512/checkpoints/017000"
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Checkpoint path exists: {checkpoint_path}")
        
        # Check for required subdirectories
        required_subdirs = ['transformer', 'vae', 'scheduler']
        for subdir in required_subdirs:
            subdir_path = os.path.join(checkpoint_path, subdir)
            if os.path.exists(subdir_path):
                print(f"✅ Found subdirectory: {subdir}")
            else:
                print(f"❌ Missing subdirectory: {subdir}")
                return False
        
        return True
    else:
        print(f"❌ Checkpoint path not found: {checkpoint_path}")
        return False

def main():
    print("🧪 Testing PartCrafter DPO Training Setup")
    print("=" * 50)
    
    tests = [
        ("Model Imports", test_model_imports),
        ("Checkpoint Path", test_checkpoint_path),
        ("Model Loading", test_model_loading),
        ("Preference Pairs Loading", test_preference_pairs_loading),
        ("Data Loading", test_data_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Test failed: {test_name}")
        except Exception as e:
            print(f"❌ Test error in {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! DPO training setup is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please fix the issues before running DPO training.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PartCrafter DPO Training Setup")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    success = main()
    sys.exit(0 if success else 1)
