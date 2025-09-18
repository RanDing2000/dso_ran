#!/usr/bin/env python3
"""
Script to find failure cases with object_size (min of length, width) in specific range and copy them to a separate directory.
Reads scene_info.txt files in failure_meshes subfolders and identifies cases with object_size in (0.0168, 0.0277].
"""

import os
import shutil
from pathlib import Path
import glob

def read_scene_info(scene_info_path):
    """
    Read scene_info.txt file and extract relevant information.
    Returns dict with scene data or None if parsing fails.
    """
    try:
        with open(scene_info_path, 'r') as f:
            lines = f.readlines()
        
        scene_data = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Scene_ID:'):
                scene_data['scene_id'] = line.split(':', 1)[1].strip()
            elif line.startswith('Num_Occluders:'):
                scene_data['num_occluders'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('Occlusion_Level:'):
                scene_data['occlusion_level'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Length:'):
                scene_data['length'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Width:'):
                scene_data['width'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Height:'):
                scene_data['height'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('Target_Name:'):
                scene_data['target_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Success:'):
                scene_data['success'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('IoU:'):
                scene_data['iou'] = float(line.split(':', 1)[1].strip())
            elif line.startswith('CD:'):
                scene_data['cd'] = float(line.split(':', 1)[1].strip())
        
        # Calculate object_size as minimum of length and width
        if 'length' in scene_data and 'width' in scene_data:
            scene_data['object_size'] = min(scene_data['length'], scene_data['width'])
        
        # Check if we have all required data
        required_fields = ['scene_id', 'length', 'width', 'height']
        if all(field in scene_data for field in required_fields):
            return scene_data
        else:
            return None
            
    except Exception as e:
        print(f"Error reading {scene_info_path}: {e}")
        return None

def copy_object_size_failures():
    """
    Main function to find and copy failure cases with object_size in specified range.
    """
    # Define paths
    failure_meshes_dir = "/home/ran.ding/projects/TARGO/targo_eval_results/vgn/eval_results/targo/targo/2025-09-11_21-17-34/visualize/failure_meshes"
    object_size_dir = "/home/ran.ding/projects/TARGO/targo_eval_results/vgn/eval_results/targo/targo/2025-09-11_21-17-34/visualize/failure_meshes_object_size"
    
    # Define object_size range: (0.0168, 0.0277]
    min_size = 0.0168
    max_size = 0.0277
    
    # Create object_size directory if it doesn't exist
    os.makedirs(object_size_dir, exist_ok=True)
    
    # Get all subdirectories in failure_meshes
    subdirs = [d for d in os.listdir(failure_meshes_dir) if os.path.isdir(os.path.join(failure_meshes_dir, d))]
    
    print(f"Found {len(subdirs)} failure case directories in {failure_meshes_dir}")
    print(f"Looking for cases with object_size in range ({min_size}, {max_size}]")
    
    # Read all scene info files and collect data
    scene_data = []
    error_cases = []
    target_range_cases = []
    
    for subdir in subdirs:
        scene_info_path = os.path.join(failure_meshes_dir, subdir, "scene_info.txt")
        
        if not os.path.exists(scene_info_path):
            print(f"Warning: scene_info.txt not found in {subdir}")
            error_cases.append(subdir)
            continue
        
        data = read_scene_info(scene_info_path)
        if data is not None:
            data['subdir'] = subdir  # Add subdirectory name for copying
            
            # Check if object_size is in target range
            if min_size < data['object_size'] <= max_size:
                target_range_cases.append(data)
            
            scene_data.append(data)
        else:
            error_cases.append(subdir)
    
    if not scene_data:
        print("No valid scene data found!")
        return
    
    print(f"\nFound {len(target_range_cases)} failure cases with object_size in range ({min_size}, {max_size}]")
    
    # Copy these cases to the new directory
    copied_count = 0
    for data in target_range_cases:
        src_path = os.path.join(failure_meshes_dir, data['subdir'])
        dst_path = os.path.join(object_size_dir, data['subdir'])
        
        try:
            if os.path.exists(dst_path):
                print(f"Warning: {data['subdir']} already exists in object_size directory, skipping")
                continue
            
            shutil.copytree(src_path, dst_path)
            print(f"Copied {data['subdir']} (Scene_ID: {data['scene_id']}, Target: {data['target_name']}, Object_Size: {data['object_size']:.6f})")
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {data['subdir']}: {e}")
    
    print(f"\nSuccessfully copied {copied_count} failure cases with object_size in range ({min_size}, {max_size}] to {object_size_dir}")
    
    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total failure cases analyzed: {len(scene_data)}")
    print(f"Error cases (parsing failed): {len(error_cases)}")
    print(f"Cases with object_size in range ({min_size}, {max_size}]: {len(target_range_cases)}")
    print(f"Successfully copied: {copied_count}")
    
    # Print detailed information about target range cases
    if target_range_cases:
        print(f"\n=== DETAILED INFORMATION FOR OBJECT_SIZE RANGE ({min_size}, {max_size}] ===")
        for i, data in enumerate(target_range_cases, 1):
            print(f"{i:3d}. {data['subdir']}")
            print(f"     Scene_ID: {data['scene_id']}")
            print(f"     Target: {data['target_name']}")
            print(f"     Object_Size: {data['object_size']:.6f} (L={data['length']:.6f}, W={data['width']:.6f})")
            print(f"     Height: {data['height']:.6f}")
            print(f"     Occlusion Level: {data['occlusion_level']:.4f}")
            print(f"     Num Occluders: {data['num_occluders']}")
            print()
    
    # Print object_size distribution
    print(f"\n=== OBJECT_SIZE DISTRIBUTION ===")
    size_ranges = [
        (0.0, 0.0168, "Very Small"),
        (0.0168, 0.0277, "Small (Target Range)"),
        (0.0277, 0.0393, "Medium-Small"),
        (0.0393, 0.0509, "Medium"),
        (0.0509, 0.0626, "Medium-Large"),
        (0.0626, 0.0742, "Large"),
        (0.0742, 1.0, "Very Large")
    ]
    
    for min_val, max_val, label in size_ranges:
        count = sum(1 for data in scene_data if min_val < data['object_size'] <= max_val)
        marker = " <-- TARGET" if min_val == min_size and max_val == max_size else ""
        print(f"{label} ({min_val:.4f}, {max_val:.4f}]: {count} cases{marker}")
    
    # Print some statistics about the target range
    if target_range_cases:
        target_sizes = [data['object_size'] for data in target_range_cases]
        print(f"\n=== TARGET RANGE STATISTICS ===")
        print(f"Min object_size in range: {min(target_sizes):.6f}")
        print(f"Max object_size in range: {max(target_sizes):.6f}")
        print(f"Average object_size in range: {sum(target_sizes)/len(target_sizes):.6f}")
        
        # Count by target names
        target_counts = {}
        for data in target_range_cases:
            target_name = data['target_name']
            target_counts[target_name] = target_counts.get(target_name, 0) + 1
        
        print(f"\n=== TARGET DISTRIBUTION IN RANGE ===")
        for target_name, count in sorted(target_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{target_name}: {count} cases")

if __name__ == "__main__":
    copy_object_size_failures()
