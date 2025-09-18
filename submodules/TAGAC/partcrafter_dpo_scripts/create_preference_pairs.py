#!/usr/bin/env python3
"""
Create preference pairs for DPO training based on penetration scores
Traverses messy_kitchen_data/dpo_data/messy_kitchen_configs and creates preference pairs
"""

import os
import json
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
from collections import defaultdict

def load_penetration_scores_for_scene(scene_dir: Path) -> Dict[str, Any]:
    """
    Load penetration scores for all seeds in a scene directory
    
    Args:
        scene_dir: Path to scene directory (e.g., 0b4113d2beec4002989365364b4e37e5_combined/)
        
    Returns:
        Dict containing penetration scores for all seeds
    """
    scene_scores = {
        'scene_name': scene_dir.name,
        'seeds': {},
        'valid_scores': []
    }
    
    # Get all seed directories (0/, 1/, 2/, etc.)
    seed_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    seed_dirs = sorted(seed_dirs, key=lambda x: int(x.name))
    
    for seed_dir in seed_dirs:
        seed = seed_dir.name
        penetration_file = seed_dir / "penetration_score.json"
        
        if penetration_file.exists():
            try:
                with open(penetration_file, 'r') as f:
                    score_data = json.load(f)
                
                if 'error' not in score_data['penetration_score']:
                    penetration_level = score_data['penetration_score']['penetration_level']
                    scene_scores['seeds'][seed] = {
                        'seed': int(seed),
                        'penetration_level': penetration_level,
                        'penetration_score': score_data['penetration_score'],
                        'seed_dir': seed_dir
                    }
                    scene_scores['valid_scores'].append(penetration_level)
                    
            except Exception as e:
                print(f"    Error loading penetration score for seed {seed}: {e}")
        else:
            print(f"    Warning: No penetration score file found for seed {seed}")
    
    return scene_scores

def create_preference_pairs_for_scene(scene_scores: Dict[str, Any], min_diff: float = 0.01) -> List[Dict[str, Any]]:
    """
    Create preference pairs for a scene based on penetration scores
    
    Args:
        scene_scores: Dict containing penetration scores for all seeds
        min_diff: Minimum difference in penetration scores to create a preference pair
        
    Returns:
        List of preference pairs
    """
    preference_pairs = []
    
    if len(scene_scores['valid_scores']) < 2:
        return preference_pairs
    
    seeds = list(scene_scores['seeds'].keys())
    penetration_levels = [scene_scores['seeds'][s]['penetration_level'] for s in seeds]
    
    # Sort seeds by penetration level (lower is better)
    sorted_indices = np.argsort(penetration_levels)
    sorted_seeds = [seeds[i] for i in sorted_indices]
    sorted_scores = [penetration_levels[i] for i in sorted_indices]
    
    # Create preference pairs: lower penetration score (win) vs higher penetration score (loss)
    for i in range(len(sorted_seeds)):
        for j in range(i + 1, len(sorted_seeds)):
            win_seed = sorted_seeds[i]
            loss_seed = sorted_seeds[j]
            win_score = sorted_scores[i]
            loss_score = sorted_scores[j]
            
            # Only create pairs if the difference is significant
            if loss_score - win_score >= min_diff:
                preference_pair = {
                    'scene_name': scene_scores['scene_name'],
                    'win_seed': int(win_seed),
                    'loss_seed': int(loss_seed),
                    'win_penetration_level': win_score,
                    'loss_penetration_level': loss_score,
                    'penetration_diff': loss_score - win_score,
                    'win_seed_dir': scene_scores['seeds'][win_seed]['seed_dir'],
                    'loss_seed_dir': scene_scores['seeds'][loss_seed]['seed_dir']
                }
                preference_pairs.append(preference_pair)
    
    return preference_pairs

def verify_preference_pair_data(win_seed_dir: Path, loss_seed_dir: Path) -> bool:
    """
    Verify that all required files exist for a preference pair
    
    Args:
        win_seed_dir: Path to winning seed directory
        loss_seed_dir: Path to losing seed directory
        
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        'cond.pt',  # Image conditioning features
        'latent_sample_000.pt',  # At least one latent feature file
        'pred_merged.glb',  # Merged mesh
        'penetration_score.json'  # Penetration score
    ]
    
    for seed_dir in [win_seed_dir, loss_seed_dir]:
        for file_name in required_files:
            file_path = seed_dir / file_name
            if not file_path.exists():
                print(f"    Warning: Missing file {file_path}")
                return False
    
    return True

def create_dpo_dataset(scene_root_dir: str, output_file: str = None, min_penetration_diff: float = 0.01) -> Dict[str, Any]:
    """
    Create DPO dataset with preference pairs from all scenes
    
    Args:
        scene_root_dir: Root directory containing scene directories
        output_file: Output file path for the DPO dataset
        min_penetration_diff: Minimum difference in penetration scores to create preference pairs
        
    Returns:
        Dict containing the DPO dataset
    """
    scene_root_path = Path(scene_root_dir)
    
    if not scene_root_path.exists():
        print(f"Error: Scene root directory not found: {scene_root_path}")
        return {}
    
    print(f"Creating DPO dataset from: {scene_root_path}")
    
    # Get all scene directories (ending with _combined)
    scene_dirs = [d for d in scene_root_path.iterdir() if d.is_dir() and d.name.endswith('_combined')]
    scene_dirs = sorted(scene_dirs)
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    dpo_dataset = {
        'dataset_info': {
            'scene_root_directory': str(scene_root_path),
            'total_scenes': len(scene_dirs),
            'min_penetration_diff': min_penetration_diff,
            'created_at': str(Path().cwd())  # Simple timestamp placeholder
        },
        'preference_pairs': [],
        'scene_summaries': {},
        'statistics': {}
    }
    
    all_penetration_scores = []
    total_pairs = 0
    valid_scenes = 0
    
    for scene_dir in scene_dirs:
        try:
            print(f"\nProcessing scene: {scene_dir.name}")
            
            # Load penetration scores for this scene
            scene_scores = load_penetration_scores_for_scene(scene_dir)
            
            if len(scene_scores['valid_scores']) < 2:
                print(f"  Skipping scene {scene_dir.name}: insufficient valid scores ({len(scene_scores['valid_scores'])})")
                continue
            
            # Create preference pairs for this scene
            preference_pairs = create_preference_pairs_for_scene(scene_scores, min_penetration_diff)
            
            # Verify and filter preference pairs
            verified_pairs = []
            for pair in preference_pairs:
                if verify_preference_pair_data(pair['win_seed_dir'], pair['loss_seed_dir']):
                    verified_pairs.append(pair)
                else:
                    print(f"    Skipping invalid preference pair: {pair['win_seed']} vs {pair['loss_seed']}")
            
            if verified_pairs:
                dpo_dataset['preference_pairs'].extend(verified_pairs)
                total_pairs += len(verified_pairs)
                valid_scenes += 1
                
                # Store scene summary
                scene_summary = {
                    'scene_name': scene_dir.name,
                    'total_seeds': len(scene_scores['seeds']),
                    'valid_scores': len(scene_scores['valid_scores']),
                    'preference_pairs': len(verified_pairs),
                    'penetration_range': [
                        float(np.min(scene_scores['valid_scores'])),
                        float(np.max(scene_scores['valid_scores']))
                    ],
                    'avg_penetration': float(np.mean(scene_scores['valid_scores']))
                }
                dpo_dataset['scene_summaries'][scene_dir.name] = scene_summary
                
                all_penetration_scores.extend(scene_scores['valid_scores'])
                
                print(f"  Created {len(verified_pairs)} preference pairs")
                print(f"  Penetration range: [{scene_summary['penetration_range'][0]:.6f}, {scene_summary['penetration_range'][1]:.6f}]")
            else:
                print(f"  No valid preference pairs created for scene {scene_dir.name}")
                
        except Exception as e:
            print(f"  Error processing scene {scene_dir.name}: {e}")
            continue
    
    # Calculate overall statistics
    if all_penetration_scores:
        dpo_dataset['statistics'] = {
            'total_scenes': len(scene_dirs),
            'valid_scenes': valid_scenes,
            'total_preference_pairs': total_pairs,
            'total_penetration_scores': len(all_penetration_scores),
            'overall_avg_penetration': float(np.mean(all_penetration_scores)),
            'overall_std_penetration': float(np.std(all_penetration_scores)),
            'overall_min_penetration': float(np.min(all_penetration_scores)),
            'overall_max_penetration': float(np.max(all_penetration_scores)),
            'avg_pairs_per_scene': total_pairs / valid_scenes if valid_scenes > 0 else 0
        }
        
        print(f"\n{'='*80}")
        print(f"DPO DATASET SUMMARY")
        print(f"{'='*80}")
        print(f"Total scenes: {dpo_dataset['statistics']['total_scenes']}")
        print(f"Valid scenes: {dpo_dataset['statistics']['valid_scenes']}")
        print(f"Total preference pairs: {dpo_dataset['statistics']['total_preference_pairs']}")
        print(f"Average pairs per scene: {dpo_dataset['statistics']['avg_pairs_per_scene']:.2f}")
        print(f"Overall penetration range: [{dpo_dataset['statistics']['overall_min_penetration']:.6f}, {dpo_dataset['statistics']['overall_max_penetration']:.6f}]")
    else:
        dpo_dataset['statistics'] = {
            'total_scenes': len(scene_dirs),
            'valid_scenes': 0,
            'total_preference_pairs': 0,
            'error': 'No valid data found'
        }
        print(f"\nWarning: No valid data found for DPO dataset creation")
    
    # Save DPO dataset
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(dpo_dataset, f, indent=2)
        print(f"\nDPO dataset saved to: {output_path}")
    
    return dpo_dataset

def main():
    parser = argparse.ArgumentParser(description='Create preference pairs for DPO training')
    parser.add_argument('--scene_root_dir', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs',
                       help='Root directory containing scene directories')
    parser.add_argument('--output_file', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json',
                       help='Output file for the DPO dataset')
    parser.add_argument('--min_penetration_diff', type=float, 
                       default=0.01,
                       help='Minimum difference in penetration scores to create preference pairs')
    
    args = parser.parse_args()
    
    # Create DPO dataset
    dpo_dataset = create_dpo_dataset(
        scene_root_dir=args.scene_root_dir,
        output_file=args.output_file,
        min_penetration_diff=args.min_penetration_diff
    )
    
    print(f"\nPreference pairs creation completed!")
    return dpo_dataset

if __name__ == "__main__":
    main()
