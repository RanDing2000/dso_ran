#!/usr/bin/env python3
"""
Minimal DPO Training Test
This script runs a very short DPO training session to verify everything works
"""

import os
import sys
import torch

# Add PartCrafter to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

def test_minimal_dpo_training():
    """Run a minimal DPO training test"""
    print("🧪 Running Minimal DPO Training Test")
    print("=" * 50)
    
    try:
        # Import training function
        sys.path.append('/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran')
        from scripts.train_partcrafter_dpo import main_training
        
        print("✅ Successfully imported training function")
        
        # Run minimal training with very small parameters
        print("🏃 Starting minimal training...")
        
        main_training(
            # Dataset parameters
            preference_pairs_file="/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json",
            max_samples=2,  # Very small dataset
            
            # Model parameters
            checkpoint_path="/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512/checkpoints/017000",
            use_lora=True,
            lora_r=8,  # Smaller LoRA rank
            lora_alpha=16,
            lora_dropout=0.1,
            
            # Training parameters
            batch_size=1,
            learning_rate=1e-4,
            scale_lr=False,
            lr_warmup_steps=2,
            use_adafactor=False,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_weight_decay=0.01,
            adam_epsilon=1e-08,
            max_train_steps=3,  # Very few steps
            max_grad_norm=1.0,
            gradient_accumulation_steps=1,
            
            # Flow matching parameters
            flow_matching_t_logit_normal_mu=0.0,
            flow_matching_t_logit_normal_sigma=1.0,
            
            # DPO parameters
            dpo_beta=0.1,
            sample_same_epsilon=True,
            
            # Logging and saving
            log_interval=1,
            save_interval=10,
            ckpt_interval=10,
            
            # General parameters
            seed=42,
            output_dir="./test_outputs",
            exp_name="minimal_dpo_test",
            logger_type="tensorboard",
            resume_from_checkpoint=None,
        )
        
        print("✅ Minimal DPO training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in minimal DPO training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_dpo_training()
    
    if success:
        print("\n🎉 Minimal DPO training test PASSED!")
        print("The training pipeline is working correctly.")
        print("You can now run full training with confidence.")
    else:
        print("\n❌ Minimal DPO training test FAILED!")
        print("Please check the error messages above.")
    
    sys.exit(0 if success else 1)
