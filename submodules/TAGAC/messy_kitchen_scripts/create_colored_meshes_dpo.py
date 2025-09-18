import os
import json
from pathlib import Path
import numpy as np
import trimesh

def create_colored_merged_mesh(pred_meshes, pred_files, seed_dir):
    """
    Create a colored version of the merged mesh with different colors for each object
    Each part number (0,1,2,3,4,5,6) gets the same color across all seeds
    
    Args:
        pred_meshes: List of individual predicted meshes
        pred_files: List of file paths corresponding to the meshes
        seed_dir: Seed directory path
    """
    try:
        if not pred_meshes:
            return
        
        print(f"      Creating colored merged mesh...")
        
        # Define 7 distinct colors (RGBA format) - one for each part number
        colors = [
            [1.0, 0.0, 0.0, 1.0],  # Red for part 0
            [0.0, 1.0, 0.0, 1.0],  # Green for part 1
            [0.0, 0.0, 1.0, 1.0],  # Blue for part 2
            [1.0, 1.0, 0.0, 1.0],  # Yellow for part 3
            [1.0, 0.0, 1.0, 1.0],  # Magenta for part 4
            [0.0, 1.0, 1.0, 1.0],  # Cyan for part 5
            [1.0, 0.5, 0.0, 1.0],  # Orange for part 6
        ]
        
        # Create colored meshes based on part number
        colored_meshes = []
        for i, (mesh, file_path) in enumerate(zip(pred_meshes, pred_files)):
            # Extract part number from filename (e.g., "pred_obj_3.ply" -> 3)
            filename = file_path.name
            if "pred_obj_" in filename and ".ply" in filename:
                try:
                    part_num = int(filename.replace("pred_obj_", "").replace(".ply", ""))
                    if 0 <= part_num < len(colors):
                        color = colors[part_num]
                    else:
                        # Fallback for unexpected part numbers
                        color = colors[part_num % len(colors)]
                except ValueError:
                    # Fallback if parsing fails
                    color = colors[i % len(colors)]
            else:
                # Fallback if filename format is unexpected
                color = colors[i % len(colors)]
            
            # Create a copy to avoid modifying original
            mesh_copy = mesh.copy()
            mesh_copy.visual.face_colors = color
            colored_meshes.append(mesh_copy)
            
            print(f"        Part {part_num if 'part_num' in locals() else i}: {filename} -> Color {color[:3]}")
        
        # Merge colored meshes
        if len(colored_meshes) > 1:
            colored_merged = trimesh.util.concatenate(colored_meshes)
        else:
            colored_merged = colored_meshes[0]
        
        # Save colored merged mesh
        colored_merged_file = seed_dir / "pred_merged_colored.ply"
        colored_merged.export(str(colored_merged_file))
        
        print(f"        Saved colored merged mesh: {colored_merged_file}")
        
        # Also save as GLB for better compatibility
        colored_merged_glb = seed_dir / "pred_merged_colored.glb"
        colored_merged.export(str(colored_merged_glb))
        
        print(f"        Saved colored merged mesh (GLB): {colored_merged_glb}")
        
    except Exception as e:
        print(f"        Error creating colored merged mesh: {e}")

def load_dpo_meshes_from_seed_dir(seed_dir):
    """
    Load predicted meshes from a DPO seed directory
    
    Args:
        seed_dir (Path): Path to seed directory (e.g., 0/, 123/, etc.)
        
    Returns:
        tuple: (pred_meshes, pred_files, merged_pred_mesh)
    """
    pred_meshes = []
    pred_files = []
    merged_pred_mesh = None
    
    try:
        # Load individual predicted meshes (pred_obj_*.ply)
        pred_files = sorted(seed_dir.glob("pred_obj_*.ply"))
        for pred_file in pred_files:
            try:
                mesh = trimesh.load_mesh(str(pred_file))
                pred_meshes.append(mesh)
                print(f"    Loaded predicted mesh: {pred_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_file.name}: {e}")
        
        # Load merged predicted mesh (pred_merged.glb)
        pred_merged_file = seed_dir / "pred_merged.glb"
        if pred_merged_file.exists():
            try:
                mesh = trimesh.load_mesh(str(pred_merged_file))
                merged_pred_mesh = mesh
                print(f"    Loaded merged predicted mesh: {pred_merged_file.name}")
            except Exception as e:
                print(f"    Error loading {pred_merged_file.name}: {e}")
        else:
            print(f"    Warning: Merged predicted mesh file not found: {pred_merged_file}")
        
        print(f"    Seed {seed_dir.name}: {len(pred_meshes)} individual pred meshes")
        print(f"      Merged pred mesh: {merged_pred_mesh is not None}")
        
    except Exception as e:
        print(f"    Error processing seed directory {seed_dir}: {e}")
    
    return pred_meshes, pred_files, merged_pred_mesh

def process_dpo_scene(scene_dir):
    """
    Process a single DPO scene directory to create colored meshes
    
    Args:
        scene_dir (Path): Path to scene directory
    """
    print(f"\nProcessing DPO scene: {scene_dir.name}")
    
    # Get all seed subdirectories
    seed_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    print(f"  Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")
    
    for seed_dir in seed_dirs:
        print(f"\n  Processing seed: {seed_dir.name}")
        
        # Check if colored mesh already exists
        colored_ply_file = seed_dir / "pred_merged_colored.ply"
        colored_glb_file = seed_dir / "pred_merged_colored.glb"
        
        if colored_ply_file.exists() and colored_glb_file.exists():
            print(f"    Colored meshes already exist, skipping...")
            continue
        
        # Load meshes for this seed
        pred_meshes, pred_files, merged_pred_mesh = load_dpo_meshes_from_seed_dir(seed_dir)
        
        if not pred_meshes:
            print(f"    Skipping seed {seed_dir.name}: No valid meshes found")
            continue
        
        # Create colored merged mesh
        create_colored_merged_mesh(pred_meshes, pred_files, seed_dir)

def create_colored_meshes_for_dpo(base_dir):
    """
    Create colored meshes for all DPO scenes
    
    Args:
        base_dir (str): Base directory containing messy_kitchen_test_100 folder
    """
    base_path = Path(base_dir)
    dpo_data_path = base_path / "messy_kitchen_test_100"
    
    if not dpo_data_path.exists():
        print(f"Error: DPO data directory not found at {dpo_data_path}")
        return
    
    print(f"Creating colored meshes for DPO data from: {dpo_data_path}")
    
    # Get all scene directories
    scene_dirs = [d for d in dpo_data_path.iterdir() if d.is_dir() and d.name.endswith('_combined')]
    print(f"Found {len(scene_dirs)} scene directories")
    
    for scene_dir in scene_dirs:
        process_dpo_scene(scene_dir)
    
    print(f"\nColored mesh creation completed for {len(scene_dirs)} scenes")

if __name__ == "__main__":
    # Path to DPO data
    base_dir = "/home/ran.ding/projects/TARGO/dpo_data"
    
    print("DPO Colored Mesh Creation")
    print("="*50)
    
    # Create colored meshes for DPO
    create_colored_meshes_for_dpo(base_dir)
    
    print(f"\nColored mesh creation completed for: {base_dir}")
