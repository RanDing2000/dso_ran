"""
Evaluation utilities for PartCrafter
"""

import os
import json
import numpy as np
import trimesh
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any
from PIL import Image

from .metric_utils import compute_chamfer_distance, compute_f_score, compute_IoU, compute_IoU_for_scene
from .data_utils import get_colored_mesh_composition


def align_meshes_with_gaps(gt_meshes: List[trimesh.Trimesh], pred_meshes: List[trimesh.Trimesh]) -> Tuple[trimesh.Scene, trimesh.Scene]:
    """
    Align predicted meshes with GT meshes using GAPS
    
    Args:
        gt_mesh: Ground Truth mesh scene
        pred_meshes: List of predicted meshes
        
    Returns:
        Tuple of (aligned_gt_scene, aligned_pred_scene)
    """
    print("Aligning meshes using GAPS...")
    
    # Check if GAPS is available
    ssr_code_path = "submodules/SSR-code"
    gaps_path = os.path.join(ssr_code_path, "external/ldif/gaps/bin/x86_64/mshalign")
    assert os.path.exists(gaps_path), "GAPS not available"    
    # Filter out empty meshes
    gt_meshes = [mesh for mesh in gt_meshes if mesh is not None and len(mesh.vertices) > 0]
    pred_meshes = [mesh for mesh in pred_meshes if mesh is not None and len(mesh.vertices) > 0]
    
    print(f"Aligning {len(gt_meshes)} GT objects with {len(pred_meshes)} predicted objects")
    
    # Create temporary directory for alignment
    with tempfile.TemporaryDirectory() as temp_dir:
        aligned_gt_meshes = []
        aligned_pred_meshes = []
        
        # For each GT object, find and align the best matching predicted object
        for i, gt_obj in enumerate(gt_meshes):
            best_cd = float('inf')
            best_pred_idx = -1
            
            # Find best matching predicted object
            for j, pred_obj in enumerate(pred_meshes):
                try:
                    cd = compute_chamfer_distance(gt_obj, pred_obj, num_samples=1000)  # Quick check
                    if cd < best_cd:
                        best_cd = cd
                        best_pred_idx = j
                except:
                    continue
            
            if best_pred_idx >= 0:
                # Save meshes for GAPS alignment
                gt_path = os.path.join(temp_dir, f"gt_{i}.ply")
                pred_path = os.path.join(temp_dir, f"pred_{i}.ply")
                aligned_path = os.path.join(temp_dir, f"aligned_{i}.ply")
                
                gt_obj.export(gt_path)
                pred_meshes[best_pred_idx].export(pred_path)
                
                # Run GAPS alignment
                try:
                    result = subprocess.run(
                        [gaps_path, gt_path, pred_path, aligned_path],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0 and os.path.exists(aligned_path):
                        # Load aligned mesh
                        aligned_mesh = trimesh.load(aligned_path)
                        aligned_gt_meshes.append(gt_obj)
                        aligned_pred_meshes.append(aligned_mesh)
                        print(f"Successfully aligned GT object {i} with predicted object {best_pred_idx}")
                    else:
                        print(f"GAPS alignment failed for GT object {i}, using original meshes")
                        aligned_gt_meshes.append(gt_obj)
                        aligned_pred_meshes.append(pred_meshes[best_pred_idx])
                        
                except Exception as e:
                    print(f"Error in GAPS alignment for GT object {i}: {e}")
                    aligned_gt_meshes.append(gt_obj)
                    aligned_pred_meshes.append(pred_meshes[best_pred_idx])
            else:
                print(f"No matching predicted object found for GT object {i}")
                aligned_gt_meshes.append(gt_obj)
        
        # Create aligned scenes
        aligned_gt_scene = trimesh.Scene(aligned_gt_meshes)
        aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
        
        print(f"Alignment completed: {len(aligned_gt_meshes)} GT objects, {len(aligned_pred_meshes)} predicted objects")
        
        return aligned_gt_scene, aligned_pred_scene


def align_merged_meshes_with_gaps(gt_merged: trimesh.Trimesh, pred_merged: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Align predicted mesh to GT mesh using GAPS
    
    Args:
        gt_merged: Merged Ground Truth mesh (reference)
        pred_merged: Merged predicted mesh (to be aligned)
        
    Returns:
        Tuple of (gt_merged, aligned_pred_merged) where pred_merged is aligned to gt_merged
    """
    print("Aligning predicted mesh to GT mesh using GAPS...")
    
    # Check if GAPS is available
    ssr_code_path = "submodules/SSR-code"
    gaps_path = os.path.join(ssr_code_path, "external/ldif/gaps/bin/x86_64/mshalign")
    if not os.path.exists(gaps_path):
        print("Warning: GAPS not available, returning original merged meshes")
        return gt_merged, pred_merged
    
    # Check if meshes are valid
    if gt_merged is None or len(gt_merged.vertices) == 0:
        print("Warning: GT merged mesh is empty")
        return gt_merged, pred_merged
    
    if pred_merged is None or len(pred_merged.vertices) == 0:
        print("Warning: Predicted merged mesh is empty")
        return gt_merged, pred_merged
    
    # Create temporary directory for alignment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save meshes for GAPS alignment
        gt_path = os.path.join(temp_dir, "gt_merged.ply")
        pred_path = os.path.join(temp_dir, "pred_merged.ply")
        aligned_pred_path = os.path.join(temp_dir, "aligned_pred_merged.ply")
        
        try:
            gt_merged.export(gt_path)
            pred_merged.export(pred_path)
            
            print(f"GT merged mesh: {len(gt_merged.vertices)} vertices, {len(gt_merged.faces)} faces")
            print(f"Pred merged mesh: {len(pred_merged.vertices)} vertices, {len(pred_merged.faces)} faces")
            
            # Run GAPS alignment (pred to gt)
            result = subprocess.run(
                [gaps_path, pred_path, gt_path, aligned_pred_path],
                # [gaps_path, gt_path, pred_path, aligned_path],
                capture_output=True,
                text=True,
                timeout=120  # Increased timeout for larger meshes
            )
            
            if result.returncode == 0 and os.path.exists(aligned_pred_path):
                # Load aligned mesh (pred aligned to gt)
                aligned_pred_merged = trimesh.load(aligned_pred_path)
                print(f"Successfully aligned predicted mesh to GT mesh")
                print(f"Aligned pred mesh: {len(aligned_pred_merged.vertices)} vertices, {len(aligned_pred_merged.faces)} faces")
                return gt_merged, aligned_pred_merged
            else:
                print(f"GAPS alignment failed for merged meshes, using original meshes")
                print(f"GAPS output: {result.stdout}")
                print(f"GAPS error: {result.stderr}")
                return gt_merged, pred_merged
                
        except Exception as e:
            print(f"Error in GAPS alignment for merged meshes: {e}")
            return gt_merged, pred_merged


def align_merged_meshes_with_gaps_and_get_transform(gt_merged: trimesh.Trimesh, pred_merged: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, np.ndarray, float]:
    """
    Align predicted mesh to GT mesh using GAPS and get transformation matrix
    
    Args:
        gt_merged: Merged Ground Truth mesh (reference)
        pred_merged: Merged predicted mesh (to be aligned)
        
    Returns:
        Tuple of (gt_merged, aligned_pred_merged, transformation_matrix, scale_factor)
        - gt_merged: GT mesh (unchanged)
        - aligned_pred_merged: Predicted mesh aligned to GT mesh
        - transformation_matrix: 4x4 homogeneous transformation matrix from GAPS
        - scale_factor: Scale factor from GAPS
    """
    print("Aligning predicted mesh to GT mesh using GAPS with transformation extraction...")
    
    # Check if GAPS is available
    ssr_code_path = "submodules/SSR-code"
    gaps_path = os.path.join(ssr_code_path, "external/ldif/gaps/bin/x86_64/mshalign")
    if not os.path.exists(gaps_path):
        print("Warning: GAPS not available, returning original merged meshes")
        return gt_merged, pred_merged, np.eye(4), 1.0
    
    # Check if meshes are valid
    if gt_merged is None or len(gt_merged.vertices) == 0:
        print("Warning: GT merged mesh is empty")
        return gt_merged, pred_merged, np.eye(4), 1.0
    
    if pred_merged is None or len(pred_merged.vertices) == 0:
        print("Warning: Predicted merged mesh is empty")
        return gt_merged, pred_merged, np.eye(4), 1.0
    
    # Create temporary directory for alignment
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save meshes for GAPS alignment
        gt_path = os.path.join(temp_dir, "gt_merged.ply")
        pred_path = os.path.join(temp_dir, "pred_merged.ply")
        aligned_pred_path = os.path.join(temp_dir, "aligned_pred_merged.ply")
        
        try:
            gt_merged.export(gt_path)
            pred_merged.export(pred_path)
            
            print(f"GT merged mesh: {len(gt_merged.vertices)} vertices, {len(gt_merged.faces)} faces")
            print(f"Pred merged mesh: {len(pred_merged.vertices)} vertices, {len(pred_merged.faces)} faces")
            
            # Run GAPS alignment (pred -> gt) with verbose output to get transformation matrix
            cmd = f"{gaps_path} {pred_path} {gt_path} {aligned_pred_path} -v"
            print(f"Running GAPS command: {cmd}")
            
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            except subprocess.TimeoutExpired:
                print("GAPS alignment timed out, using original meshes")
                return gt_merged, pred_merged, np.eye(4), 1.0
            
            if result.returncode == 0 and os.path.exists(aligned_pred_path):
                # Load aligned mesh (pred aligned to gt)
                aligned_pred_merged = trimesh.load(aligned_pred_path)
                print(f"Successfully aligned predicted mesh to GT mesh")
                print(f"Aligned pred mesh: {len(aligned_pred_merged.vertices)} vertices, {len(aligned_pred_merged.faces)} faces")
                
                # Parse transformation matrix and scale from GAPS output
                transformation_matrix, scale_factor = parse_gaps_transformation(result.stdout)
                
                # If parsing failed, try to extract transformation from mesh comparison
                if np.allclose(transformation_matrix, np.eye(4)) and scale_factor == 1.0:
                    print("Warning: GAPS output parsing failed, attempting to extract transformation from mesh comparison")
                    try:
                        # Use ICP-like method to extract transformation from pred_merged to aligned_pred_merged
                        transformation_matrix, scale_factor = extract_transformation_from_meshes(pred_merged, aligned_pred_merged)
                        print(f"Extracted transformation from mesh comparison: scale={scale_factor}")
                    except Exception as e:
                        print(f"Failed to extract transformation from mesh comparison: {e}")
                        print("Using identity transformation")
                        transformation_matrix = np.eye(4)
                        scale_factor = 1.0
                
                return gt_merged, aligned_pred_merged, transformation_matrix, scale_factor
            else:
                print(f"GAPS alignment failed for merged meshes, using original meshes")
                print(f"GAPS output: {result.stdout}")
                print(f"GAPS error: {result.stderr}")
                return gt_merged, pred_merged, np.eye(4), 1.0
                
        except Exception as e:
            print(f"Error in GAPS alignment for merged meshes: {e}")
            return gt_merged, pred_merged, np.eye(4), 1.0


from typing import Tuple
import numpy as np
import re

def parse_gaps_transformation(gaps_output: str) -> Tuple[np.ndarray, float]:
    """
    Parse 4x4 homogeneous transform and scale factor from GAPS verbose stdout.

    Supports lines like:
      Matrix[0][0-3] = r00 r01 r02 t0
      ...
      Matrix[3][0-3] = 0 0 0 1
      Scale = s

    Falls back to parsing 4 bare numeric rows if needed.
    """
    T = np.eye(4, dtype=np.float64)
    scale = 1.0

    lines = gaps_output.splitlines()

    # --- 1) Preferred: "Matrix[i][0-3] = a b c d"
    row_pat = re.compile(r'^Matrix\[(\d+)\]\[0-3\]\s*=\s*([-+0-9eE\.\s]+)$')
    rows_found = {}

    for raw in lines:
        line = raw.strip()
        m = row_pat.match(line)
        if not m:
            continue
        i = int(m.group(1))
        try:
            vals = [float(x) for x in m.group(2).split()]
        except ValueError:
            continue
        if 0 <= i <= 3 and len(vals) == 4:
            rows_found[i] = vals

    if len(rows_found) == 4:
        for i in range(4):
            T[i, :4] = rows_found[i]

    # --- 2) Fallback: 4 bare numeric rows (e.g., "a b c d")
    if len(rows_found) < 4:
        bare_rows = []
        bare_pat = re.compile(r'^[\d\-\+\.eE\s]+$')
        for raw in lines:
            line = raw.strip()
            if bare_pat.match(line):
                nums = line.split()
                if len(nums) == 4:
                    try:
                        bare_rows.append([float(x) for x in nums])
                    except ValueError:
                        pass
        if len(bare_rows) >= 4:
            T = np.array(bare_rows[:4], dtype=np.float64)

    # --- 3) Scale: "Scale = s" or "ScaleFactor = s"
    scale_pat = re.compile(r'^\s*Scale(?:Factor)?\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$', re.I)
    for raw in lines:
        line = raw.strip()
        m = scale_pat.match(line)
        if m:
            try:
                scale = float(m.group(1))
                break
            except ValueError:
                pass

    # --- 4) Last resort: infer scale from det(R) if scale still 1.0 and matrix not identity-ish
    if (scale == 1.0) and not np.allclose(T, np.eye(4), atol=1e-8):
        det3 = np.linalg.det(T[:3, :3])
        if det3 != 0:
            scale = np.cbrt(abs(det3))  # use abs to avoid reflection sign

    # Ensure last row is [0,0,0,1] if stdout omitted it
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    return T, scale


# def parse_gaps_transformation(gaps_output: str) -> Tuple[np.ndarray, float]:
#     """
#     Parse transformation matrix and scale factor from GAPS verbose output
    
#     Args:
#         gaps_output: GAPS stdout output with -v flag
        
#     Returns:
#         Tuple of (transformation_matrix, scale_factor)
#         transformation_matrix: 4x4 homogeneous transformation matrix
#         scale_factor: scale factor from GAPS
#     """
#     import re
    
#     # Initialize default values
#     transformation_matrix = np.eye(4, dtype=np.float64)
#     scale_factor = 1.0
    
#     try:
#         lines = gaps_output.split('\n')
        
#         # Look for transformation matrix (4x4 matrix)
#         matrix_lines = []
#         in_matrix = False
        
#         for line in lines:
#             line = line.strip()
            
#             # Look for matrix start (usually contains numbers in scientific notation)
#             if re.match(r'^[\d\-\+\.eE\s]+$', line) and len(line.split()) == 4:
#                 matrix_lines.append(line)
#                 in_matrix = True
#             elif in_matrix and line == '':
#                 # Empty line after matrix indicates end
#                 break
#             elif in_matrix and not re.match(r'^[\d\-\+\.eE\s]+$', line):
#                 # Non-matrix line, stop parsing
#                 break
        
#         # Parse the 4x4 transformation matrix
#         if len(matrix_lines) >= 4:
#             matrix_data = []
#             for line in matrix_lines[:4]:  # Take first 4 lines
#                 row = [float(x) for x in line.split()]
#                 if len(row) == 4:
#                     matrix_data.append(row)
            
#             if len(matrix_data) == 4:
#                 transformation_matrix = np.array(matrix_data, dtype=np.float64)
#                 print(f"Parsed GAPS transformation matrix:\n{transformation_matrix}")
        
#         # Look for scale factor
#         for line in lines:
#             line = line.strip()
#             # Look for scale factor in various formats
#             if 'ScaleFactor' in line or 'scale' in line.lower():
#                 # Extract number from line
#                 numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
#                 if numbers:
#                     scale_factor = float(numbers[0])
#                     print(f"Parsed GAPS scale factor: {scale_factor}")
#                     break
        
#         # Alternative: extract scale from transformation matrix determinant
#         if scale_factor == 1.0:
#             det_3x3 = np.linalg.det(transformation_matrix[:3, :3])
#             if det_3x3 > 0:
#                 scale_factor = np.cbrt(det_3x3)
#                 print(f"Extracted scale factor from matrix determinant: {scale_factor}")
        
#     except Exception as e:
#         print(f"Error parsing GAPS transformation: {e}")
#         print(f"GAPS output:\n{gaps_output}")
    
    return transformation_matrix, scale_factor


def compute_aligned_metrics(gt_meshes: List[trimesh.Trimesh], pred_meshes: List[trimesh.Trimesh], 
                          num_samples: int = 10000) -> Dict[str, Any]:
    """
    Compute evaluation metrics for multi-object reconstruction
    
    Args:
        gt_meshes: List of Ground Truth meshes (reference)
        pred_meshes: List of predicted meshes (already aligned if using GAPS)
        num_samples: Number of sample points for metric computation
        
    Returns:
        Dictionary containing various metrics and aligned scenes
    """
    print(f"GT scene contains {len(gt_meshes)} objects")
    print(f"Predicted scene contains {len(pred_meshes)} objects")
    
    
    if len(gt_meshes) == 0 or len(pred_meshes) == 0:
        print("Warning: No valid meshes found")
        return {
            'chamfer_distance': float('inf'),
            'f_score': 0.0,
            'iou': 0.0,
            'chamfer_distance_std': 0.0,
            'f_score_std': 0.0,
            'iou_std': 0.0,
            'per_object_cds': [],
            'per_object_fscores': [],
            'per_object_ious': [],
            'scene_iou': 0.0
        }
    
    # Use the provided meshes directly (alignment already done in eval.py)
    aligned_pred_meshes = pred_meshes
    
    # Step 4: Compute per-object metrics
    print("Step 4: Computing per-object metrics...")
    per_object_cds = []
    per_object_fscores = []
    per_object_ious = []
    
    # For each predicted object, find best matching GT object and compute metrics
    for i, pred_mesh in enumerate(aligned_pred_meshes):
        best_cd = float('inf')
        best_fscore = 0.0
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_mesh in enumerate(gt_meshes):
            try:
                # Compute Chamfer distance
                cd = compute_chamfer_distance(gt_mesh, pred_mesh, num_samples=num_samples//len(gt_meshes))

                # gt_mesh.export(os.path.join(f"gt_{j}.ply"))
                gt_mesh.export(f"gt_{j}.ply", include_attributes=False)
                pred_mesh.export(f"pred_{i}.ply", include_attributes=False)

                # Compute F-score
                fscore = compute_f_score(gt_mesh, pred_mesh, num_samples=num_samples//len(gt_meshes), threshold=0.1)
                # Compute IoU (using smaller grid for speed)
                try:
                    iou = compute_IoU(gt_mesh, pred_mesh, num_grids=16, scale=1.5)
                except Exception as e:
                    print(f"Error computing IoU: {e}")
                    iou = 0.0
                
                if cd < best_cd:
                    best_cd = cd
                    best_fscore = fscore
                    best_iou = iou
                    best_gt_idx = j
                    
            except Exception as e:
                print(f"Error computing metrics for pred object {i} and GT object {j}: {e}")
                continue
        
        if best_gt_idx >= 0:
            per_object_cds.append(best_cd)
            per_object_fscores.append(best_fscore)
            per_object_ious.append(best_iou)
            print(f"Pred object {i} matched with GT object {best_gt_idx}: CD={best_cd:.6f}, F-score={best_fscore:.6f}, IoU={best_iou:.6f}")
        else:
            per_object_cds.append(float('inf'))
            per_object_fscores.append(0.0)
            per_object_ious.append(0.0)
            print(f"No match found for pred object {i}")
    
    # Compute scene-level metrics
    if per_object_cds:
        scene_cd = np.mean(per_object_cds)
        scene_fscore = np.mean(per_object_fscores)
        scene_iou = np.mean(per_object_ious)
        scene_cd_std = np.std(per_object_cds)
        scene_fscore_std = np.std(per_object_fscores)
        scene_iou_std = np.std(per_object_ious)
    else:
        scene_cd = float('inf')
        scene_fscore = 0.0
        scene_iou = 0.0
        scene_cd_std = 0.0
        scene_fscore_std = 0.0
        scene_iou_std = 0.0
    
    # Create metrics dictionary
    metrics = {
        'chamfer_distance': float(scene_cd),
        'f_score': float(scene_fscore),
        'iou': float(scene_iou),
        'chamfer_distance_std': float(scene_cd_std),
        'f_score_std': float(scene_fscore_std),
        'iou_std': float(scene_iou_std),
        'per_object_cds': per_object_cds,
        'per_object_fscores': per_object_fscores,
        'per_object_ious': per_object_ious,
        'scene_iou': float(scene_iou)
    }
    
    # Create aligned scenes for compatibility
    aligned_gt_scene = trimesh.Scene(gt_meshes)
    aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
    
    # Store aligned scenes for later use
    metrics['aligned_gt_scene'] = aligned_gt_scene
    metrics['aligned_pred_scene'] = aligned_pred_scene
    
    # Create merged meshes for compatibility
    if len(gt_meshes) > 1:
        metrics['aligned_gt_merged'] = trimesh.util.concatenate(gt_meshes)
    else:
        metrics['aligned_gt_merged'] = gt_meshes[0] if gt_meshes else None
    
    if len(aligned_pred_meshes) > 1:
        metrics['aligned_pred_merged'] = trimesh.util.concatenate(aligned_pred_meshes)
    else:
        metrics['aligned_pred_merged'] = aligned_pred_meshes[0] if aligned_pred_meshes else None
    
    print(f"Final scene metrics: CD={metrics['chamfer_distance']:.6f}±{metrics['chamfer_distance_std']:.6f}, "
          f"F-score={metrics['f_score']:.6f}±{metrics['f_score_std']:.6f}, "
          f"IoU={metrics['iou']:.6f}±{metrics['iou_std']:.6f}")
    
    return metrics


def save_meshes_with_alignment(pred_meshes: List[trimesh.Trimesh], gt_mesh: trimesh.Scene,
                              output_dir: str, case_name: str,
                              aligned_gt_scene: trimesh.Scene = None,
                              aligned_pred_scene: trimesh.Scene = None,
                              metrics: Dict = None) -> str:
    """
    Save predicted and GT meshes as GLB files with alignment support
    
    Args:
        pred_meshes: List of predicted meshes
        gt_mesh: Ground Truth mesh/scene
        output_dir: Output directory
        case_name: Case name for file naming
        aligned_gt_scene: Aligned GT scene
        aligned_pred_scene: Aligned predicted scene
        
    Returns:
        Path to the mesh directory
    """
    mesh_dir = os.path.join(output_dir, case_name)
    os.makedirs(mesh_dir, exist_ok=True)
    
    # Save original predicted meshes
    print(f"Saving {len(pred_meshes)} predicted meshes...")
    for i, mesh in enumerate(pred_meshes):
        if mesh is not None and len(mesh.vertices) > 0:
            filename = f"pred_part_{i:02d}.glb"
            filepath = os.path.join(mesh_dir, filename)
            mesh.export(filepath)
            print(f"  Saved {filename}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Save merged predicted mesh
    if len(pred_meshes) > 1:
        try:
            merged_mesh = get_colored_mesh_composition(pred_meshes)
            merged_path = os.path.join(mesh_dir, "pred_merged.glb")
            merged_mesh.export(merged_path)
            print(f"  Saved pred_merged.glb: {len(merged_mesh.vertices)} vertices, {len(merged_mesh.faces)} faces")
        except Exception as e:
            print(f"  Error saving merged mesh: {e}")
    elif len(pred_meshes) == 1 and pred_meshes[0] is not None:
        import shutil
        shutil.copy2(
            os.path.join(mesh_dir, "pred_part_00.glb"),
            os.path.join(mesh_dir, "pred_merged.glb")
        )
    
    # Save GT meshes
    print(f"Saving GT meshes...")
    if isinstance(gt_mesh, trimesh.Scene):
        gt_meshes = gt_mesh.dump()
        print(f"  GT scene contains {len(gt_meshes)} geometries")
        
        for i, mesh in enumerate(gt_meshes):
            if mesh is not None and len(mesh.vertices) > 0:
                filename = f"gt_part_{i:02d}.glb"
                filepath = os.path.join(mesh_dir, filename)
                mesh.export(filepath)
                print(f"  Saved {filename}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Save merged GT mesh
        if len(gt_meshes) > 1:
            try:
                gt_merged = get_colored_mesh_composition(gt_meshes)
                gt_merged_path = os.path.join(mesh_dir, "gt_merged.glb")
                gt_merged.export(gt_merged_path)
                print(f"  Saved gt_merged.glb: {len(gt_merged.vertices)} vertices, {len(gt_merged.faces)} faces")
            except Exception as e:
                print(f"  Error saving GT merged mesh: {e}")
        elif len(gt_meshes) == 1 and gt_meshes[0] is not None:
            import shutil
            shutil.copy2(
                os.path.join(mesh_dir, "gt_part_00.glb"),
                os.path.join(mesh_dir, "gt_merged.glb")
            )
    else:
        # Single GT mesh
        if gt_mesh is not None and len(gt_mesh.vertices) > 0:
            gt_path = os.path.join(mesh_dir, "gt.glb")
            gt_mesh.export(gt_path)
            print(f"  Saved gt.glb: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
            
            # Also save as gt_merged for consistency
            gt_merged_path = os.path.join(mesh_dir, "gt_merged.glb")
            gt_mesh.export(gt_merged_path)
    
    # Save aligned meshes if available
    if aligned_gt_scene is not None and aligned_pred_scene is not None:
        print("Saving aligned meshes...")
        
        # Save aligned GT scene
        aligned_gt_path = os.path.join(mesh_dir, "gt_aligned.glb")
        aligned_gt_scene.export(aligned_gt_path)
        print(f"  Saved gt_aligned.glb")
        
        # Save aligned predicted scene
        aligned_pred_path = os.path.join(mesh_dir, "pred_aligned.glb")
        aligned_pred_scene.export(aligned_pred_path)
        print(f"  Saved pred_aligned.glb")
        
        # Save merged aligned meshes if available (from compute_aligned_metrics)
        if metrics is not None and 'aligned_gt_merged' in metrics and 'aligned_pred_merged' in metrics:
            # This is a merged mesh approach, save the merged aligned meshes
            try:
                aligned_gt_merged = metrics['aligned_gt_merged']
                aligned_pred_merged = metrics['aligned_pred_merged']
                
                # Save merged aligned meshes
                gt_merged_aligned_path = os.path.join(mesh_dir, "gt_merged_aligned.glb")
                aligned_gt_merged.export(gt_merged_aligned_path)
                print(f"  Saved gt_merged_aligned.glb: {len(aligned_gt_merged.vertices)} vertices, {len(aligned_gt_merged.faces)} faces")
                
                pred_merged_aligned_path = os.path.join(mesh_dir, "pred_merged_aligned.glb")
                aligned_pred_merged.export(pred_merged_aligned_path)
                print(f"  Saved pred_merged_aligned.glb: {len(aligned_pred_merged.vertices)} vertices, {len(aligned_pred_merged.faces)} faces")
                
            except Exception as e:
                print(f"  Error saving merged aligned meshes: {e}")
        else:
            # Original approach: save individual aligned objects
            aligned_gt_meshes = aligned_gt_scene.dump()
            aligned_pred_meshes = aligned_pred_scene.dump()
            
            for i, (gt_mesh, pred_mesh) in enumerate(zip(aligned_gt_meshes, aligned_pred_meshes)):
                if gt_mesh is not None and len(gt_mesh.vertices) > 0:
                    gt_aligned_path = os.path.join(mesh_dir, f"gt_aligned_part_{i:02d}.glb")
                    gt_mesh.export(gt_aligned_path)
                    print(f"  Saved gt_aligned_part_{i:02d}.glb: {len(gt_mesh.vertices)} vertices, {len(gt_mesh.faces)} faces")
                
                if pred_mesh is not None and len(pred_mesh.vertices) > 0:
                    pred_aligned_path = os.path.join(mesh_dir, f"pred_aligned_part_{i:02d}.glb")
                    pred_mesh.export(pred_aligned_path)
                    print(f"  Saved pred_aligned_part_{i:02d}.glb: {len(pred_mesh.vertices)} vertices, {len(pred_mesh.faces)} faces")
    
    # Create and save summary
    summary_data = create_mesh_summary(case_name, pred_meshes, gt_mesh)
    summary_path = os.path.join(mesh_dir, "mesh_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Mesh summary saved to: {summary_path}")
    return mesh_dir


def create_mesh_summary(case_name: str, pred_meshes: List[trimesh.Trimesh], gt_mesh: trimesh.Scene) -> Dict:
    """Create a summary file with mesh statistics"""
    summary_data = {
        'case_name': case_name,
        'predicted_meshes': {
            'count': len(pred_meshes),
            'parts': []
        },
        'gt_meshes': {
            'count': len(gt_mesh.dump()) if isinstance(gt_mesh, trimesh.Scene) else 1,
            'parts': []
        }
    }
    
    # Add predicted mesh info
    for i, mesh in enumerate(pred_meshes):
        if mesh is not None and len(mesh.vertices) > 0:
            summary_data['predicted_meshes']['parts'].append({
                'index': i,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'filename': f"pred_part_{i:02d}.glb"
            })
    
    # Add GT mesh info
    if isinstance(gt_mesh, trimesh.Scene):
        gt_meshes = gt_mesh.dump()
        for i, mesh in enumerate(gt_meshes):
            if mesh is not None and len(mesh.vertices) > 0:
                summary_data['gt_meshes']['parts'].append({
                    'index': i,
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.faces),
                    'filename': f"gt_part_{i:02d}.glb"
                })
    else:
        if gt_mesh is not None and len(gt_mesh.vertices) > 0:
            summary_data['gt_meshes']['parts'].append({
                'index': 0,
                'vertices': len(gt_mesh.vertices),
                'faces': len(gt_mesh.faces),
                'filename': "gt.glb"
            })
    
    return summary_data


def setup_gaps_tools() -> bool:
    """
    Setup GAPS tools for evaluation
    
    Returns:
        True if GAPS is available, False otherwise
    """
    print("Setting up GAPS tools for evaluation...")
    
    # Check if SSR code exists (for GAPS building)
    ssr_code_path = "submodules/SSR-code"
    if not os.path.exists(ssr_code_path):
        print(f"Warning: SSR code not found at {ssr_code_path}")
        print("GAPS tools will not be available for evaluation")
        return False
    
    # Check if GAPS is already built
    gaps_path = os.path.join(ssr_code_path, "external/ldif/gaps/bin/x86_64/mshalign")
    if os.path.exists(gaps_path):
        print("GAPS tools already installed, skipping build step")
        print("GAPS tools setup completed")
        return True
    
    # If GAPS not found, try to build it
    print("GAPS tools not found, attempting to build...")
    print("Note: This may require sudo privileges for system dependencies")
    try:
        # Try building without sudo first
        result = subprocess.run(
            ["bash", "build_gaps.sh"],
            cwd=os.path.join(ssr_code_path, "external"),
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("GAPS built successfully")
            return True
        else:
            print(f"GAPS build failed: {result.stderr}")
            print("GAPS tools will not be available for evaluation")
            print("You can manually build GAPS later if needed:")
            print(f"  cd {os.path.join(ssr_code_path, 'external')}")
            print("  sudo bash build_gaps.sh")
            return False
            
    except subprocess.TimeoutExpired:
        print("GAPS build timed out")
        print("GAPS tools will not be available for evaluation")
        return False
    except Exception as e:
        print(f"Error building GAPS: {e}")
        print("GAPS tools will not be available for evaluation")
        return False


# def extract_and_apply_transformation(original_mesh: trimesh.Trimesh, aligned_mesh: trimesh.Trimesh, 
#                                    target_meshes: List[trimesh.Trimesh]) -> List[trimesh.Trimesh]:
#     """
#     Extract transformation from GAPS alignment and apply to individual predicted meshes
    
#     Args:
#         original_mesh: Original merged predicted mesh
#         aligned_mesh: Aligned merged predicted mesh (aligned to GT)
#         target_meshes: List of individual predicted meshes to transform
        
#     Returns:
#         List of transformed individual predicted meshes
#     """
#     try:
#         from scipy.spatial.transform import Rotation
#         import numpy as np
#         from scipy.optimize import minimize
        
#         # Sample points from both meshes
#         num_samples = min(10000, len(original_mesh.vertices))
#         original_points = original_mesh.sample(num_samples)
#         aligned_points = aligned_mesh.sample(num_samples)
        
#         # Find the transformation that maps original predicted points to aligned predicted points
#         # We'll use ICP-like approach to find the best transformation
        
#         # Center the points
#         original_center = np.mean(original_points, axis=0)
#         aligned_center = np.mean(aligned_points, axis=0)
        
#         original_centered = original_points - original_center
#         aligned_centered = aligned_points - aligned_center
        
#         # Find rotation using SVD
#         H = original_centered.T @ aligned_centered
#         U, S, Vt = np.linalg.svd(H)
#         R = Vt.T @ U.T
        
#         # Ensure proper rotation matrix
#         if np.linalg.det(R) < 0:
#             Vt[-1, :] *= -1
#             R = Vt.T @ U.T
        
#         # Find scale
#         scale = np.trace(H @ R) / np.trace(original_centered.T @ original_centered)
        
#         # Find translation
#         translation = aligned_center - scale * R @ original_center
        
#         print(f"Extracted transformation (pred to GT): scale={scale:.4f}, translation={translation}")
        
#         # Apply transformation to individual predicted meshes
#         transformed_meshes = []
#         for mesh in target_meshes:
#             transformed_mesh = mesh.copy()
            
#             # Apply scale, rotation, and translation
#             transformed_mesh.vertices = scale * (R @ transformed_mesh.vertices.T).T + translation
            
#             transformed_meshes.append(transformed_mesh)
        
#         return transformed_meshes
        
#     except Exception as e:
#         print(f"Error in transformation extraction: {e}")
#         # Fallback: use simple translation
#         try:
#             original_center = np.mean(original_mesh.vertices, axis=0)
#             aligned_center = np.mean(aligned_mesh.vertices, axis=0)
#             translation = aligned_center - original_center
            
#             transformed_meshes = []
#             for mesh in target_meshes:
#                 transformed_mesh = mesh.copy()
#                 transformed_mesh.vertices += translation
#                 transformed_meshes.append(transformed_mesh)
            
#             print(f"Applied simple translation (pred to GT): {translation}")
#             return transformed_meshes
            
#         except Exception as e2:
#             print(f"Error in fallback transformation: {e2}")
#             return target_meshes.copy()


# from __future__ import annotations
from typing import List
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def _umeyama_similarity(X: np.ndarray, Y: np.ndarray, with_scale: bool = True):
    """
    Umeyama (1991): 求相似变换 (s, R, t)，使 Y ≈ s * R @ X + t
    X, Y: (N, 3) 且已建立一一对应（例如通过最近邻或已知对应）
    返回:
        s: float, 均匀尺度
        R: (3,3) 旋转矩阵（det≈+1）
        t: (3,) 平移
    """
    assert X.shape == Y.shape and X.shape[1] == 3
    N = X.shape[0]
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    # 协方差
    Sigma = (Xc.T @ Yc) / N
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    varX = (Xc ** 2).sum() / max(N, 1)
    if with_scale:
        s = np.trace(np.diag(D) @ S) / max(varX, 1e-12)
    else:
        s = 1.0

    t = muY - s * (R @ muX)
    return s, R, t


def _to_homo_matrix(s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """构造 4x4 齐次矩阵，对应均匀尺度+旋转+平移。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def extract_and_apply_transformation(
    original_mesh: trimesh.Trimesh,
    aligned_mesh: trimesh.Trimesh,
    target_meshes: List[trimesh.Trimesh],
    *,
    num_samples: int = 8000,
    iters: int = 15,
    trim_ratio: float = 0.8,
    with_scale: bool = True,
    random_state: int | None = 0,
) -> List[trimesh.Trimesh]:
    """
    从 (original_mesh -> aligned_mesh) 估计全局相似变换 T，然后把 T 应用到每个 target_mesh。

    参数:
        original_mesh : 合并前的预测 mesh（对齐前）
        aligned_mesh  : 经过外部对齐（如 GAPS）后的合并预测 mesh（已对齐到 GT 坐标系）
        target_meshes : 需要应用同一变换的单个预测 mesh 列表
        num_samples   : 采样点数上限
        iters         : ICP 迭代次数
        trim_ratio    : 修剪比例（0~1），保留较近的对应对，增强鲁棒性
        with_scale    : 是否估计尺度
        random_state  : 采样随机种子，便于复现

    返回:
        变换后的 target_meshes 副本列表
    """
    if random_state is not None:
        np.random.seed(random_state)

    # 1) 采样表面点（表面均匀采样，比直接用 vertices 更鲁棒）
    n0 = min(num_samples, max(len(original_mesh.vertices), 1000))
    n1 = min(num_samples, max(len(aligned_mesh.vertices), 1000))
    X = original_mesh.sample(n0)  # (N,3)
    Y = aligned_mesh.sample(n1)   # (M,3)

    # 2) ICP 初始化
    T_global = np.eye(4, dtype=np.float64)
    X_curr = X.copy()
    tree = cKDTree(Y)

    for _ in range(max(iters, 1)):
        dists, idx = tree.query(X_curr, k=1, workers=-1)
        Y_nn = Y[idx]

        # trimmed ICP: 丢掉最差的 (1 - trim_ratio) 对应
        if 0.0 < trim_ratio < 1.0:
            thr = np.quantile(dists, trim_ratio)
            mask = dists <= thr
            X_used = X_curr[mask]
            Y_used = Y_nn[mask]
        else:
            X_used, Y_used = X_curr, Y_nn

        if len(X_used) < 10:
            # 对应太少，提前停止
            break

        s, R, t = _umeyama_similarity(X_used, Y_used, with_scale=with_scale)
        T_step = _to_homo_matrix(s, R, t)
        T_global = T_step @ T_global

        # 更新 X_curr
        X_h = np.c_[X, np.ones((X.shape[0], 1))]
        X_curr = (T_global @ X_h.T).T[:, :3]

        # 简单收敛判据
        rot_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        if np.linalg.norm(t) < 1e-5 and (abs(s - 1.0) < 1e-5) and rot_angle < 0.01:
            break

    # 3) 打印/检查结果
    s_final = np.cbrt(max(np.linalg.det(T_global[:3, :3]), 1e-24))
    R_final = T_global[:3, :3] / s_final
    t_final = T_global[:3, 3]
    # R_final 数值误差处理
    if np.linalg.det(R_final) < 0:
        # 罕见：若因数值误差导致 det<0，做一次修正
        U, _, Vt = np.linalg.svd(R_final)
        S = np.eye(3); S[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        R_final = U @ S @ Vt
        T_global[:3, :3] = s_final * R_final
    print(f"[ICP] Estimated transform: scale={s_final:.6f}, det(R)={np.linalg.det(R_final):.6f}, t={t_final}")

    # 4) 应用到每个 target mesh（返回副本）
    out: List[trimesh.Trimesh] = []
    for m in target_meshes:
        m2 = m.copy()
        m2.apply_transform(T_global)  # 通过齐次矩阵一次性应用 s、R、t，并维护内部缓存
        out.append(m2)
    return out


def apply_gaps_transformation_to_meshes(meshes: List[trimesh.Trimesh], transformation_matrix: np.ndarray) -> List[trimesh.Trimesh]:
    """
    Apply GAPS transformation matrix to individual meshes
    
    Args:
        meshes: List of individual meshes to transform
        transformation_matrix: 4x4 homogeneous transformation matrix from GAPS
        
    Returns:
        List of transformed meshes
    """
    transformed_meshes = []
    
    for i, mesh in enumerate(meshes):
        transformed_mesh = mesh.copy()
        
        # Apply the transformation matrix directly
        transformed_mesh.apply_transform(transformation_matrix)
        
        transformed_meshes.append(transformed_mesh)
        print(f"Applied GAPS transformation to mesh {i}")
    
    return transformed_meshes


def extract_transformation_from_meshes(original_mesh: trimesh.Trimesh, aligned_mesh: trimesh.Trimesh) -> Tuple[np.ndarray, float]:
    """
    Extract transformation matrix and scale factor from original mesh to aligned mesh
    
    Args:
        original_mesh: Original mesh (before alignment)
        aligned_mesh: Aligned mesh (after alignment)
        
    Returns:
        Tuple of (transformation_matrix, scale_factor)
        transformation_matrix: 4x4 homogeneous transformation matrix
        scale_factor: scale factor
    """
    try:
        # Sample points from both meshes
        num_samples = min(10000, len(original_mesh.vertices), len(aligned_mesh.vertices))
        original_points = original_mesh.sample(num_samples)
        aligned_points = aligned_mesh.sample(num_samples)
        
        # Use Umeyama algorithm to find transformation
        s, R, t = _umeyama_similarity(original_points, aligned_points, with_scale=True)
        
        # Create 4x4 homogeneous transformation matrix
        transformation_matrix = _to_homo_matrix(s, R, t)
        
        print(f"Extracted transformation: scale={s:.6f}, translation={t}")
        print(f"Transformation matrix:\n{transformation_matrix}")
        
        return transformation_matrix, s
        
    except Exception as e:
        print(f"Error extracting transformation from meshes: {e}")
        # Return identity transformation as fallback
        return np.eye(4), 1.0
