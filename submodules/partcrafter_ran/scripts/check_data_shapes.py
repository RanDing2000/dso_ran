#!/usr/bin/env python3
"""
Check data shapes for DPO training
"""

import torch
import os
import json

def check_data_shapes():
    """Check the shapes of latent and cond features"""
    
    # Load preference pairs
    preference_pairs_file = "/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json"
    with open(preference_pairs_file, 'r') as f:
        data = json.load(f)
    
    preference_pairs = data['preference_pairs']
    
    # Check first few pairs
    for i, pair in enumerate(preference_pairs[:3]):
        print(f"\n=== Pair {i+1} ===")
        print(f"Scene: {pair['scene_name']}")
        print(f"Win seed: {pair['win_seed']}, Loss seed: {pair['loss_seed']}")
        
        # Check win seed data
        win_seed_dir = pair['win_seed_dir']
        win_cond_path = os.path.join(win_seed_dir, 'cond.pt')
        
        if os.path.exists(win_cond_path):
            cond_features = torch.load(win_cond_path, map_location='cpu')
            print(f"Win cond shape: {cond_features.shape}")
            
            # Check latent features
            latent_files = [f for f in os.listdir(win_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
            if latent_files:
                latent_path = os.path.join(win_seed_dir, latent_files[0])
                latent_features = torch.load(latent_path, map_location='cpu')
                print(f"Win latent shape: {latent_features.shape}")
                
                # Calculate num_parts
                # Assuming the first dimension represents tokens per part
                # We need to figure out how many parts this represents
                print(f"Latent tokens: {latent_features.shape[0]}")
                
        # Check loss seed data
        loss_seed_dir = pair['loss_seed_dir']
        loss_cond_path = os.path.join(loss_seed_dir, 'cond.pt')
        
        if os.path.exists(loss_cond_path):
            cond_features = torch.load(loss_cond_path, map_location='cpu')
            print(f"Loss cond shape: {cond_features.shape}")
            
            # Check latent features
            latent_files = [f for f in os.listdir(loss_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
            if latent_files:
                latent_path = os.path.join(loss_seed_dir, latent_files[0])
                latent_features = torch.load(latent_path, map_location='cpu')
                print(f"Loss latent shape: {latent_features.shape}")

if __name__ == "__main__":
    check_data_shapes()
