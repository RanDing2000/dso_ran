#!/usr/bin/env python3
"""
PSR (Penetration Surface Ratio) evaluation script
Reads aligned_pred_merged.glb files from evaluation results and computes penetration metrics
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import trimesh
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

def _build_pq_list(meshes: List[trimesh.Trimesh]):
    """预构建并复用 ProximityQuery（关键）"""
    return [trimesh.proximity.ProximityQuery(m) for m in meshes]

def _aabb_overlap(a_min, a_max, b_min, b_max, pad=0.0):
    """对称 AABB 交叠测试（带 padding）"""
    a_min_pad = a_min - pad
    a_max_pad = a_max + pad
    b_min_pad = b_min - pad
    b_max_pad = b_max + pad
    return not (
        (a_max_pad < b_min_pad).any() or
        (a_min_pad > b_max_pad).any()
    )

def penetration_sdf_surface_fast(
    meshes: List[trimesh.Trimesh],
    samples_per_mesh: int = 4000,
    seed: int = 0,
    aggregate: str = "max",
    aabb_pad: float = 0.0,
    dtype = np.float32,
    eps_rel: float = 1e-6,
    max_batch_points: int = 200_000
) -> Dict[str, Any]:
    """
    Fast multi-object penetration via surface sampling + cross SDF checks.
    关键优化：对每个 j 一次性计算所有相关 i 的点的 SDF，减少函数调用次数。
    Speed-ups: reuse ProximityQuery, AABB broad-phase culling, vectorized updates, batch processing.
    Returns PSR, AD_all, AD_inside and per-object detail.
    """
    assert len(meshes) >= 2, "At least two meshes are required"
    rng = np.random.default_rng(seed)

    # 统一到 float32，减少拷贝和开销
    meshes = [m.astype(dtype) if hasattr(m, 'astype') else m for m in meshes]

    # 预采样（表面均匀采样）
    sampled_pts: List[np.ndarray] = []
    for m in meshes:
        pts = m.sample(samples_per_mesh)
        sampled_pts.append(np.asarray(pts, dtype=dtype))

    num_meshes = len(meshes)

    # 预构建 PQ（最关键的复用）
    pqs = _build_pq_list(meshes)

    # AABB 预取
    bounds = np.stack([np.vstack([m.bounds[0], m.bounds[1]]) for m in meshes], axis=0)  # (M, 2, 3)
    bounds_min_all = bounds[:, 0, :]
    bounds_max_all = bounds[:, 1, :]

    # 场景尺度
    bounds_min = bounds_min_all.min(axis=0)
    bounds_max = bounds_max_all.max(axis=0)
    scene_diag = float(np.linalg.norm(bounds_max - bounds_min)) + 1e-12

    total_points = 0
    total_inside = 0
    total_negdepth_sum = 0.0

    per_object: List[Dict[str, Any]] = []
    pairwise_inside_ratio = np.zeros((num_meshes, num_meshes), dtype=dtype)

    # 主循环：按物体 i 的采样点，和潜在相交的 j 做 SDF
    for i in range(num_meshes):
        pts_i = sampled_pts[i]
        n_points = len(pts_i)

        inside_any = np.zeros(n_points, dtype=bool)
        neg_depth_acc = np.zeros(n_points, dtype=dtype)

        # 广相位：只保留 AABB 交叠的 j
        cand_js = []
        a_min, a_max = bounds_min_all[i], bounds_max_all[i]
        for j in range(num_meshes):
            if i == j:
                continue
            if _aabb_overlap(a_min, a_max, bounds_min_all[j], bounds_max_all[j], pad=aabb_pad):
                cand_js.append(j)
            else:
                pairwise_inside_ratio[i, j] = 0.0  # 明确置零

        # 对候选 j 做 SDF
        for j in cand_js:
            # 使用已复用的 PQ（避免每次重建）
            d_ij = pqs[j].signed_distance(pts_i)  # numpy array, +为外、-为内
            # 向量化：把负值的深度一次性累到 neg_depth_acc
            neg_add = np.clip(-d_ij, 0, None).astype(dtype)

            if aggregate == "max":
                # 取穿插深度的最大值
                neg_depth_acc = np.maximum(neg_depth_acc, neg_add)
            else:
                # 累加（更重）
                neg_depth_acc += neg_add

            mask_in = d_ij < 0
            if mask_in.any():
                inside_any |= mask_in
                pairwise_inside_ratio[i, j] = float(mask_in.mean())
            else:
                pairwise_inside_ratio[i, j] = 0.0

        # 汇总 i 的指标
        cnt_in = int(inside_any.sum())
        total_points += n_points
        total_inside += cnt_in
        total_negdepth_sum += float(neg_depth_acc.sum())

        avg_depth_inside = float(neg_depth_acc[inside_any].mean()) if cnt_in > 0 else 0.0
        per_object.append({
            "surface_points": n_points,
            "inside_count": cnt_in,
            "inside_ratio": cnt_in / max(n_points, 1),
            "avg_depth_inside": avg_depth_inside,
            "avg_depth_inside_norm_by_diag": (avg_depth_inside / scene_diag) if cnt_in > 0 else 0.0,
        })

    # psr = total_inside / max(total_points, 1)
    psr = 1 - (total_inside / max(total_points, 1))
    print(f"PSR: {psr}")
    ad_all = total_negdepth_sum / max(total_points, 1)
    ad_inside = (total_negdepth_sum / max(total_inside, 1)) if total_inside > 0 else 0.0

    return {
        "PSR": psr,
        "AD_all": ad_all,
        "AD_inside": ad_inside,
        "AD_all_norm_by_diag": ad_all / scene_diag,
        "AD_inside_norm_by_diag": ad_inside / scene_diag if ad_inside > 0 else 0.0,
        "per_object": per_object,
        "pairwise_inside_ratio": pairwise_inside_ratio,
        "scene_diag": scene_diag,
        "samples_per_mesh": samples_per_mesh,
        "used_aabb_pad": aabb_pad,
        "dtype": str(dtype),
    }


def load_meshes_from_glb(glb_path: str) -> List[trimesh.Trimesh]:
    scene = trimesh.load(glb_path, process=False)
    meshes = []
    # Get raw geometries (still in local coordinates)
    for name, geom in scene.geometry.items():
        # print(f"{name}: {type(geom)}, v={len(geom.vertices)}, f={len(geom.faces)}")
        meshes.append(geom)
    return meshes


def load_meshes_from_aligned_ply_files(case_dir: str) -> List[trimesh.Trimesh]:
    """
    Load meshes from aligned_pred_obj_*.ply files in a case directory
    
    Args:
        case_dir: Path to the case directory containing aligned_pred_obj_*.ply files
        
    Returns:
        List of trimesh.Trimesh objects
    """
    meshes = []
    
    if not os.path.exists(case_dir):
        print(f"Case directory does not exist: {case_dir}")
        return meshes
    
    # Find all aligned_pred_obj_*.ply files
    i = 0
    while True:
        ply_filename = f"aligned_pred_obj_{i}.ply"
        ply_path = os.path.join(case_dir, ply_filename)
        
        if os.path.exists(ply_path):
            try:
                mesh = trimesh.load(ply_path, process=False)
                if isinstance(mesh, trimesh.Trimesh):
                    meshes.append(mesh)
                    print(f"  Loaded: {ply_filename} (vertices: {len(mesh.vertices)}, faces: {len(mesh.faces)})")
                else:
                    print(f"  [WARN] {ply_filename} is not a valid mesh")
            except Exception as e:
                print(f"  [WARN] Failed to load {ply_filename}: {e}")
            i += 1
        else:
            break
    
    print(f"Loaded {len(meshes)} meshes from aligned_pred_obj_*.ply files in {case_dir}")
    return meshes

def simplify_mesh_for_collision(mesh: trimesh.Trimesh, target_faces: int = 50000) -> trimesh.Trimesh:
    """
    为碰撞检测创建低面数版本的网格
    
    Args:
        mesh: 原始网格
        target_faces: 目标面数
        
    Returns:
        简化后的网格
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    
    try:
        # 尝试使用 trimesh 的简化功能
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        if simplified is not None and len(simplified.faces) > 0:
            print(f"  Simplified mesh: {len(mesh.faces)} -> {len(simplified.faces)} faces")
            return simplified
    except Exception as e:
        print(f"  [WARN] Trimesh simplification failed: {e}")
    
    # 如果简化失败，返回原网格
    print(f"  [WARN] Using original mesh with {len(mesh.faces)} faces")
    return mesh


def load_meshes_from_aligned_ply_files_optimized(case_dir: str, simplify_for_collision: bool = True, target_faces: int = 50000) -> List[trimesh.Trimesh]:
    """
    加载并优化网格（可选简化以提高碰撞检测性能）
    
    Args:
        case_dir: 案例目录路径
        simplify_for_collision: 是否简化网格以提高性能
        target_faces: 简化目标面数
        
    Returns:
        优化后的网格列表
    """
    meshes = load_meshes_from_aligned_ply_files(case_dir)
    
    if simplify_for_collision and meshes:
        print(f"Optimizing {len(meshes)} meshes for collision detection...")
        optimized_meshes = []
        for i, mesh in enumerate(meshes):
            if simplify_for_collision:
                optimized_mesh = simplify_mesh_for_collision(mesh, target_faces)
            else:
                optimized_mesh = mesh
            optimized_meshes.append(optimized_mesh)
        
        # 验证简化后的网格
        total_original_faces = sum(len(m.faces) for m in meshes)
        total_optimized_faces = sum(len(m.faces) for m in optimized_meshes)
        print(f"  Total faces: {total_original_faces} -> {total_optimized_faces} (reduction: {total_original_faces/total_optimized_faces:.1f}x)")
        
        return optimized_meshes
    
    return meshes


def make_grid(bounds_min, bounds_max, resolution=128, pad_ratio=0.01):
    """Create a regular grid of points for penetration calculation"""
    size = bounds_max - bounds_min
    pad = pad_ratio * np.max(size)
    lo = bounds_min - pad
    hi = bounds_max + pad
    xs = np.linspace(lo[0], hi[0], resolution, endpoint=False)
    ys = np.linspace(lo[1], hi[1], resolution, endpoint=False)
    zs = np.linspace(lo[2], hi[2], resolution, endpoint=False)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return P, (resolution, resolution, resolution)

def penetration_rate_from_points(meshes, points, chunk=200_000):
    """
    Calculate penetration rate from grid points
    
    Args:
        meshes: List of trimesh.Trimesh objects
        points: Grid points to check
        chunk: Chunk size for memory efficiency
        
    Returns:
        PR: Penetration rate (0~1, higher means more severe penetration)
        U: Number of union points
        S: Sum of individual occupied points
        pair_matrix: Pairwise overlap matrix
    """
    if len(meshes) == 0:
        return 0.0, 0, 0, None
    
    # occupancy_counts[x] = 有多少个 mesh 包含该点
    occ = np.zeros(len(points), dtype=np.int32)
    
    print(f"Processing {len(meshes)} meshes with {len(points)} grid points (chunk size: {chunk})")

    for i, m in enumerate(meshes):
        try:
            # 分块以省内存
            inside = np.zeros(len(points), dtype=bool)
            for j in range(0, len(points), chunk):
                end = min(j+chunk, len(points))
                # 优先 fast winding；若结果不稳定，可改用 signed_distance<0
                inside[j:end] = m.contains(points[j:end])
            occ += inside.astype(np.int32)
            print(f"  Mesh {i}: {inside.sum()} points inside")
        except Exception as e:
            print(f"  [WARN] Failed to process mesh {i}: {e}")
            # 如果某个mesh失败，继续处理其他mesh
            continue

    S = int(occ.sum())                    # Σ N_i
    U = int((occ > 0).sum())              # N_union
    PR = 1.0 - (U / max(S, 1))            # 穿插率（防零）
    
    print(f"Total occupied points: {S}, Union points: {U}, Penetration rate: {PR:.4f}")

    # 额外：返回 pairwise 矩阵，快速定位"谁和谁"穿插最多
    pair_matrix = None
    if len(meshes) <= 20:  # 太多就跳过，避免内存暴涨
        try:
            per_mesh_occ = []
            for i, m in enumerate(meshes):
                try:
                    inside = np.zeros(len(points), dtype=bool)
                    for j in range(0, len(points), chunk):
                        end = min(j+chunk, len(points))
                        inside[j:end] = m.contains(points[j:end])
                    per_mesh_occ.append(inside)
                except Exception as e:
                    print(f"  [WARN] Failed to create pairwise matrix for mesh {i}: {e}")
                    # 如果某个mesh失败，用全零向量代替
                    per_mesh_occ.append(np.zeros(len(points), dtype=bool))
            
            if len(per_mesh_occ) > 0:
                per_mesh_occ = np.stack(per_mesh_occ, axis=1)  # [N_points, M]
                M = per_mesh_occ.shape[1]
                pair_matrix = np.zeros((M, M), dtype=np.int64)
                for a in range(M):
                    for b in range(a+1, M):
                        pair_matrix[a, b] = np.logical_and(per_mesh_occ[:, a], per_mesh_occ[:, b]).sum()
                print(f"Created pairwise overlap matrix: {M}x{M}")
        except Exception as e:
            print(f"  [WARN] Failed to create pairwise matrix: {e}")
            pair_matrix = None

    return PR, U, S, pair_matrix

def compute_penetration_rate_from_meshes(meshes, resolution=128):
    """
    Compute penetration rate from a list of meshes
    
    Args:
        meshes: List of trimesh.Trimesh objects
        resolution: Grid resolution for sampling
        
    Returns:
        Dictionary with penetration metrics
    """
    assert len(meshes) > 0, "No meshes provided."
    
    # 过滤掉无效的网格
    valid_meshes = []
    for i, m in enumerate(meshes):
        if m is not None and hasattr(m, 'bounds') and hasattr(m, 'contains'):
            try:
                # 检查网格是否有效
                if m.vertices.shape[0] > 0 and m.faces.shape[0] > 0:
                    valid_meshes.append(m)
                else:
                    print(f"[WARN] Mesh {i} has no vertices or faces, skipping")
            except Exception as e:
                print(f"[WARN] Mesh {i} validation failed: {e}, skipping")
        else:
            print(f"[WARN] Mesh {i} is invalid or missing required attributes, skipping")
    
    if len(valid_meshes) < 2:
        print(f"[WARN] Only {len(valid_meshes)} valid meshes found, cannot compute penetration rate")
        return {
            "penetration_rate": 0.0,
            "union_points": 0,
            "sum_individual_points": 0,
            "resolution": resolution,
            "pairwise_overlap_matrix": None,
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes)
        }
    
    print(f"Computing penetration rate for {len(valid_meshes)} valid meshes (from {len(meshes)} total)")
    
    try:
        # 计算边界框
        bounds = np.array([m.bounds for m in valid_meshes])  # [M, 2, 3]
        lo = bounds[:,0,:].min(axis=0)
        hi = bounds[:,1,:].max(axis=0)
        
        # 动态调整分辨率以避免内存问题
        scene_size = np.max(hi - lo)
        if scene_size > 10.0:  # 如果场景太大，降低分辨率
            adjusted_resolution = max(32, resolution // 2)
            print(f"[INFO] Large scene detected (size: {scene_size:.2f}), reducing resolution from {resolution} to {adjusted_resolution}")
            resolution = adjusted_resolution
        
        points, shape = make_grid(lo, hi, resolution=resolution)
        print(f"Created grid with {len(points)} points, shape: {shape}")
        
        PR, U, S, pair_mat = penetration_rate_from_points(valid_meshes, points)
        
        return {
            "penetration_rate": PR,                 # 0~1，越大穿插越严重
            "union_points": int(U),
            "sum_individual_points": int(S),
            "resolution": resolution,
            "pairwise_overlap_matrix": pair_mat,    # 可选，用于定位问题
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes),
            "grid_points": len(points),
            "scene_bounds": {
                "min": lo.tolist(),
                "max": hi.tolist(),
                "size": (hi - lo).tolist()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to compute penetration rate: {e}")
        return {
            "penetration_rate": 0.0,
            "union_points": 0,
            "sum_individual_points": 0,
            "resolution": resolution,
            "pairwise_overlap_matrix": None,
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes),
            "error": str(e)
        }

def penetration_volume_ratio(meshes: List[trimesh.Trimesh], n_samples: int = 200_000, seed: int = 0) -> Dict[str, Any]:
    """
    估计: 1 - Vol(union)/sum Vol(M_i)
    体积均匀采样（同一总体） -> 稳定、尺度不变
    
    Args:
        meshes: List of trimesh.Trimesh objects
        n_samples: Number of volume sampling points
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with volume overlap metrics
    """
    assert len(meshes) > 0, "No meshes provided."
    
    # 过滤掉无效的网格
    valid_meshes = []
    for i, m in enumerate(meshes):
        if m is not None and hasattr(m, 'bounds') and hasattr(m, 'contains'):
            try:
                # 检查网格是否有效
                if m.vertices.shape[0] > 0 and m.faces.shape[0] > 0:
                    valid_meshes.append(m)
                else:
                    print(f"[WARN] Mesh {i} has no vertices or faces, skipping")
            except Exception as e:
                print(f"[WARN] Mesh {i} validation failed: {e}, skipping")
        else:
            print(f"[WARN] Mesh {i} is invalid or missing required attributes, skipping")
    
    if len(valid_meshes) < 2:
        print(f"[WARN] Only {len(valid_meshes)} valid meshes found, cannot compute volume overlap")
        return {
            "penetration_score": 0.0,
            "vol_union_est": 0.0,
            "vol_sum_est": 0.0,
            "n_samples": n_samples,
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes),
            "error": "Insufficient valid meshes"
        }
    
    print(f"Computing volume overlap for {len(valid_meshes)} valid meshes (from {len(meshes)} total)")
    
    try:
        rng = np.random.default_rng(seed)
        
        # 统一场景包围盒
        bounds = np.array([m.bounds for m in valid_meshes])  # [M, 2, 3]
        mins = bounds[:, 0, :].min(axis=0)
        maxs = bounds[:, 1, :].max(axis=0)
        box_size = maxs - mins
        
        # 体积采样
        u = rng.random((n_samples, 3), dtype=np.float32)
        pts = mins + u * box_size
        
        # inside 测试（使用稳健方法）
        eps = 1e-6 * float(np.linalg.norm(box_size))
        inside_mat = []
        
        print(f"Testing {n_samples} points against {len(valid_meshes)} meshes...")
        
        for i, m in enumerate(valid_meshes):
            try:
                # 使用ProximityQuery的signed_distance
                pq = trimesh.proximity.ProximityQuery(m)
                d = pq.signed_distance(pts)
                inside_mat.append(d < -eps)
                print(f"  Mesh {i}: {inside_mat[-1].sum()} points inside")
            except Exception as e:
                print(f"  [WARN] Failed to process mesh {i}: {e}")
                # 如果失败，用全False代替
                inside_mat.append(np.zeros(n_samples, dtype=bool))
        
        if not inside_mat:
            raise ValueError("No valid inside tests completed")
        
        inside_mat = np.stack(inside_mat, axis=1)  # (N, M) bool
        
        # 估计体积（蒙特卡洛）：P(inside) * Vol(box)
        vol_box = float(box_size.prod())
        p_union = inside_mat.any(1).mean()
        p_sum = inside_mat.mean(0).sum()  # sum_i P(inside_i)
        
        vol_union_est = p_union * vol_box
        vol_sum_est = p_sum * vol_box
        
        # 计算渗透分数：1 - Vol(union)/sum Vol(M_i)
        score = 1.0 - (vol_union_est / (vol_sum_est + 1e-12))
        score = max(0.0, min(1.0, score))  # 限制在[0,1]范围内
        
        return {
            "penetration_score": float(score),
            "vol_union_est": vol_union_est,
            "vol_sum_est": vol_sum_est,
            "n_samples": n_samples,
            "box_diag": float(np.linalg.norm(box_size)),
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes),
            "eps_threshold": eps,
            "scene_bounds": {
                "min": mins.tolist(),
                "max": maxs.tolist(),
                "size": box_size.tolist()
            }
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to compute volume overlap: {e}")
        import traceback
        traceback.print_exc()
        return {
            "penetration_score": 0.0,
            "vol_union_est": 0.0,
            "vol_sum_est": 0.0,
            "n_samples": n_samples,
            "valid_mesh_count": len(valid_meshes),
            "total_mesh_count": len(meshes),
            "error": str(e)
        }

def evaluate_single_psr(case_dir: str, case_name: str, samples_per_mesh: int = 4000, 
                       grid_resolution: int = 64, verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate PSR for a single case directory containing aligned_pred_obj_*.ply files
    
    Args:
        case_dir: Path to the case directory containing aligned_pred_obj_*.ply files
        case_name: Name of the test case
        samples_per_mesh: Number of surface samples per mesh for SDF-based PSR
        grid_resolution: Grid resolution for volume-based penetration rate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing PSR evaluation results
    """
    print(f"\nEvaluating PSR for case: {case_name}")
    print(f"Case directory: {case_dir}")
    
    try:
        # Load meshes from aligned_pred_obj_*.ply files
        meshes = load_meshes_from_aligned_ply_files_optimized(case_dir)
        
        if len(meshes) < 2:
            print(f"[WARN] Case {case_name}: Only {len(meshes)} meshes found, skipping PSR calculation")
            return {
                'case_name': case_name,
                'case_dir': case_dir,
                'num_meshes': len(meshes),
                'error': 'Insufficient meshes for PSR calculation',
                'success': False
            }
        
        print(f"Loaded {len(meshes)} meshes from {case_dir}")
        
        # Method 1: SDF-based penetration (surface sampling)
        print(f"Computing SDF-based PSR with {samples_per_mesh} samples per mesh...")
        sdf_results = penetration_sdf_surface_fast(
            meshes,
            samples_per_mesh=samples_per_mesh,
            seed=42,
            aggregate="max",
            aabb_pad=0.01,
            eps_rel=1e-6,
            max_batch_points=200_000
        )
        
        # Method 2: Grid-based penetration rate (volume sampling)
        print(f"Computing grid-based penetration rate with resolution {grid_resolution}...")
        grid_results = compute_penetration_rate_from_meshes(meshes, resolution=grid_resolution)
        
        # Method 3: Volume overlap ratio (new metric)
        print(f"Computing volume overlap ratio...")
        volume_results = penetration_volume_ratio(meshes, n_samples=100000, seed=42)
        
        # Combine results
        results = {
            'case_name': case_name,
            'case_dir': case_dir,
            'num_meshes': len(meshes),
            'success': True,
            'sdf_based': {
                'PSR': sdf_results['PSR'],
                'AD_all': sdf_results['AD_all'],
                'AD_inside': sdf_results['AD_inside'],
                'AD_all_norm_by_diag': sdf_results['AD_all_norm_by_diag'],
                'AD_inside_norm_by_diag': sdf_results['AD_inside_norm_by_diag'],
                'per_object': sdf_results['per_object'],
                'pairwise_inside_ratio': sdf_results['pairwise_inside_ratio'].tolist() if isinstance(sdf_results['pairwise_inside_ratio'], np.ndarray) else sdf_results['pairwise_inside_ratio'],
                'scene_diag': sdf_results['scene_diag'],
                'samples_per_mesh': sdf_results['samples_per_mesh']
            },
            'grid_based': grid_results,
            'volume_based': volume_results,
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"PSR evaluation completed for {case_name}")
        print(f"  SDF-based PSR: {sdf_results['PSR']:.6f}")
        print(f"  Grid-based penetration rate: {grid_results['penetration_rate']:.6f}")
        print(f"  Volume overlap score: {volume_results['penetration_score']:.6f}")
        
        return results
        
    except Exception as e:
        print(f"Error evaluating PSR for case {case_name}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'case_name': case_name,
            'case_dir': case_dir,
            'error': str(e),
            'success': False
        }

def save_psr_results_to_txt(case_name: str, results: Dict[str, Any], output_path: str):
    """
    Save PSR results to a readable text file
    
    Args:
        case_name: Name of the test case
        results: Dictionary containing PSR evaluation results
        output_path: Path to save the PSR results file
    """
    with open(output_path, 'w') as f:
        f.write(f"PartCrafter PSR Evaluation Results\n")
        f.write(f"Case: {case_name}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 50 + "\n\n")
        
        if not results.get('success', False):
            f.write(f"EVALUATION FAILED:\n")
            f.write(f"  Error: {results.get('error', 'Unknown error')}\n")
            return
        
        # Basic info
        f.write("BASIC INFO:\n")
        f.write(f"  Case directory: {results['case_dir']}\n")
        f.write(f"  Number of meshes: {results['num_meshes']}\n")
        f.write(f"  Evaluation time: {results['evaluation_time']}\n")
        f.write("\n")
        
        # SDF-based results
        sdf = results['sdf_based']
        f.write("SDF-BASED PENETRATION METRICS:\n")
        f.write(f"  PSR (Penetration Surface Ratio): {sdf['PSR']:.6f}\n")
        f.write(f"  AD_all (Average Depth - all points): {sdf['AD_all']:.6f}\n")
        f.write(f"  AD_inside (Average Depth - inside points): {sdf['AD_inside']:.6f}\n")
        f.write(f"  AD_all (normalized): {sdf['AD_all_norm_by_diag']:.6f}\n")
        f.write(f"  AD_inside (normalized): {sdf['AD_inside_norm_by_diag']:.6f}\n")
        f.write(f"  Scene diagonal: {sdf['scene_diag']:.6f}\n")
        f.write(f"  Samples per mesh: {sdf['samples_per_mesh']}\n")
        f.write("\n")
        
        # Per-object details
        if 'per_object' in sdf and sdf['per_object']:
            f.write("PER-OBJECT SDF METRICS:\n")
            for i, obj in enumerate(sdf['per_object']):
                f.write(f"  Object {i+1}:\n")
                f.write(f"    Surface points: {obj['surface_points']}\n")
                f.write(f"    Inside count: {obj['inside_count']}\n")
                f.write(f"    Inside ratio: {obj['inside_ratio']:.6f}\n")
                f.write(f"    Avg depth inside: {obj['avg_depth_inside']:.6f}\n")
                f.write(f"    Avg depth inside (normalized): {obj['avg_depth_inside_norm_by_diag']:.6f}\n")
                f.write("\n")
        
        # Grid-based results
        grid = results['grid_based']
        f.write("GRID-BASED PENETRATION METRICS:\n")
        f.write(f"  Penetration rate: {grid['penetration_rate']:.6f}\n")
        f.write(f"  Union points: {grid['union_points']}\n")
        f.write(f"  Sum individual points: {grid['sum_individual_points']}\n")
        f.write(f"  Grid resolution: {grid['resolution']}\n")
        f.write(f"  Valid meshes: {grid['valid_mesh_count']}/{grid['total_mesh_count']}\n")
        f.write(f"  Grid points: {grid['grid_points']}\n")
        f.write("\n")
        
        # Volume-based results (new)
        if 'volume_based' in results:
            volume = results['volume_based']
            f.write("VOLUME-BASED OVERLAP METRICS:\n")
            f.write(f"  Penetration score: {volume['penetration_score']:.6f}\n")
            f.write(f"  Volume union estimate: {volume['vol_union_est']:.6f}\n")
            f.write(f"  Volume sum estimate: {volume['vol_sum_est']:.6f}\n")
            f.write(f"  Volume samples: {volume['n_samples']}\n")
            f.write(f"  Box diagonal: {volume['box_diag']:.6f}\n")
            f.write(f"  Valid meshes: {volume['valid_mesh_count']}/{volume['total_mesh_count']}\n")
            f.write("\n")
            
            # Scene bounds from volume results
            if 'scene_bounds' in volume:
                bounds = volume['scene_bounds']
                f.write("VOLUME SCENE BOUNDS:\n")
                f.write(f"  Min: {bounds['min']}\n")
                f.write(f"  Max: {bounds['max']}\n")
                f.write(f"  Size: {bounds['size']}\n")
                f.write("\n")
        
        # Scene bounds (from grid results as fallback)
        if 'scene_bounds' in grid and 'volume_based' not in results:
            bounds = grid['scene_bounds']
            f.write("SCENE BOUNDS:\n")
            f.write(f"  Min: {bounds['min']}\n")
            f.write(f"  Max: {bounds['max']}\n")
            f.write(f"  Size: {bounds['size']}\n")
            f.write("\n")
        
        f.write("EVALUATION COMPLETED SUCCESSFULLY\n")

def find_case_dirs_with_aligned_ply_files(evaluation_dir: str) -> List[Tuple[str, str]]:
    """
    Find all case directories containing aligned_pred_obj_*.ply files
    
    Args:
        evaluation_dir: Path to the evaluation results directory
        
    Returns:
        List of tuples: (case_name, case_dir)
    """
    case_dirs = []
    
    if not os.path.exists(evaluation_dir):
        print(f"Evaluation directory does not exist: {evaluation_dir}")
        return case_dirs
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(evaluation_dir):
        # Check if this directory contains any aligned_pred_obj_*.ply files
        has_aligned_ply = any(file.startswith("aligned_pred_obj_") and file.endswith(".ply") for file in files)
        
        if has_aligned_ply:
            case_name = os.path.basename(root)
            case_dir = root
            case_dirs.append((case_name, case_dir))
            print(f"  Found case: {case_name} at {case_dir}")
    
    print(f"Found {len(case_dirs)} case directories with aligned_pred_obj_*.ply files")
    return case_dirs

def main():
    parser = argparse.ArgumentParser(description='PSR evaluation script for PartCrafter results')
    parser.add_argument('--evaluation_dir', type=str, 
                       default='./results/evaluation_objects_demo_test_100',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/psr_evaluation',
                       help='Output directory for PSR results')
    parser.add_argument('--samples_per_mesh', type=int, default=4000,
                       help='Number of surface samples per mesh for SDF-based PSR')
    parser.add_argument('--grid_resolution', type=int, default=64,
                       help='Grid resolution for volume-based penetration rate')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Find all case directories containing aligned_pred_obj_*.ply files
    case_dirs = find_case_dirs_with_aligned_ply_files(args.evaluation_dir)
    
    if not case_dirs:
        print("No case directories with aligned_pred_obj_*.ply files found. Exiting.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate PSR for each case
    all_results = []
    successful_results = []
    failed_results = []
    
    print(f"\nStarting PSR evaluation for {len(case_dirs)} cases...")
    
    for case_name, case_dir in tqdm(case_dirs, desc="PSR evaluation progress"):
        # Evaluate PSR
        results = evaluate_single_psr(
            case_dir=case_dir,
            case_name=case_name,
            samples_per_mesh=args.samples_per_mesh,
            grid_resolution=args.grid_resolution,
            verbose=args.verbose
        )
        
        all_results.append(results)
        
        if results['success']:
            successful_results.append(results)
            
            # Save individual PSR results to text file
            case_output_dir = os.path.join(args.output_dir, case_name)
            os.makedirs(case_output_dir, exist_ok=True)
            
            psr_txt_path = os.path.join(case_output_dir, "psr_results.txt")
            save_psr_results_to_txt(case_name, results, psr_txt_path)
            print(f"PSR results saved to: {psr_txt_path}")
        else:
            failed_results.append(results)
    
    # Save summary results
    summary_path = os.path.join(args.output_dir, "psr_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"PartCrafter PSR Evaluation Summary\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total cases: {len(all_results)}\n")
        f.write(f"  Successful cases: {len(successful_results)}\n")
        f.write(f"  Failed cases: {len(failed_results)}\n")
        f.write(f"  Success rate: {len(successful_results)/len(all_results)*100:.1f}%\n")
        f.write("\n")
        
        if successful_results:
            # Calculate summary metrics
            all_psrs = [r['sdf_based']['PSR'] for r in successful_results]
            all_ad_alls = [r['sdf_based']['AD_all'] for r in successful_results]
            all_ad_insides = [r['sdf_based']['AD_inside'] for r in successful_results]
            all_grid_prs = [r['grid_based']['penetration_rate'] for r in successful_results]
            
            f.write("SUMMARY METRICS:\n")
            f.write(f"  PSR (Penetration Surface Ratio):\n")
            f.write(f"    Mean: {np.mean(all_psrs):.6f}\n")
            f.write(f"    Std: {np.std(all_psrs):.6f}\n")
            f.write(f"    Min: {np.min(all_psrs):.6f}\n")
            f.write(f"    Max: {np.max(all_psrs):.6f}\n")
            f.write("\n")
            
            f.write(f"  AD_all (Average Depth - all points):\n")
            f.write(f"    Mean: {np.mean(all_ad_alls):.6f}\n")
            f.write(f"    Std: {np.std(all_ad_alls):.6f}\n")
            f.write(f"    Min: {np.min(all_ad_alls):.6f}\n")
            f.write(f"    Max: {np.max(all_ad_alls):.6f}\n")
            f.write("\n")
            
            f.write(f"  AD_inside (Average Depth - inside points):\n")
            f.write(f"    Mean: {np.mean(all_ad_insides):.6f}\n")
            f.write(f"    Std: {np.std(all_ad_insides):.6f}\n")
            f.write(f"    Min: {np.min(all_ad_insides):.6f}\n")
            f.write(f"    Max: {np.max(all_ad_insides):.6f}\n")
            f.write("\n")
            
            f.write(f"  Grid-based Penetration Rate:\n")
            f.write(f"    Mean: {np.mean(all_grid_prs):.6f}\n")
            f.write(f"    Std: {np.std(all_grid_prs):.6f}\n")
            f.write(f"    Min: {np.min(all_grid_prs):.6f}\n")
            f.write(f"    Max: {np.max(all_grid_prs):.6f}\n")
            f.write("\n")
        
        # Failed cases
        if failed_results:
            f.write("FAILED CASES:\n")
            for result in failed_results:
                f.write(f"  {result['case_name']}: {result.get('error', 'Unknown error')}\n")
            f.write("\n")
        
        f.write("PSR EVALUATION COMPLETED\n")
        f.write(f"Results saved to: {args.output_dir}\n")
    
    # Save detailed results to JSON
    json_path = os.path.join(args.output_dir, "psr_evaluation_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'evaluation_dir': args.evaluation_dir,
            'total_cases': len(all_results),
            'successful_cases': len(successful_results),
            'failed_cases': len(failed_results),
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'samples_per_mesh': args.samples_per_mesh,
                'grid_resolution': args.grid_resolution
            },
            'detailed_results': all_results
        }, f, indent=2)
    
    # Create CSV summary
    if successful_results:
        df_data = []
        for result in successful_results:
            df_data.append({
                'case_name': result['case_name'],
                'num_meshes': result['num_meshes'],
                'psr': result['sdf_based']['PSR'],
                'ad_all': result['sdf_based']['AD_all'],
                'ad_inside': result['sdf_based']['AD_inside'],
                'ad_all_norm': result['sdf_based']['AD_all_norm_by_diag'],
                'ad_inside_norm': result['sdf_based']['AD_inside_norm_by_diag'],
                'grid_penetration_rate': result['grid_based']['penetration_rate'],
                'grid_resolution': result['grid_based']['resolution'],
                'valid_meshes': result['grid_based']['valid_mesh_count']
            })
        
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(args.output_dir, "psr_evaluation_results.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV summary saved to: {csv_path}")
    
    print(f"\nPSR evaluation completed!")
    print(f"Summary saved to: {summary_path}")
    print(f"Detailed results saved to: {json_path}")
    print(f"All results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
