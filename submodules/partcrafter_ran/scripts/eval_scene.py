# #!/usr/bin/env python3
# """
# Large-scale test script: Evaluate PartCrafter performance on messy kitchen dataset
# """

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
import trimesh
from PIL import Image
from typing import Dict, List, Tuple, Any
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import yaml
# from src.vgn.utils import create_tsdf
# from src.vgn.simulation import ClutterRemovalSim
# from src.vgn.utils import Camera

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from src.models.autoencoders import TripoSGVAEModel
from src.models.transformers import PartCrafterDiTModel
from src.schedulers import RectifiedFlowScheduler
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from src.utils.eval_utils import (
    setup_gaps_tools, compute_aligned_metrics, 
    save_meshes_with_alignment, align_merged_meshes_with_gaps_and_get_transform,
    apply_gaps_transformation_to_meshes
)
from src.utils.visualization_utils import (
    render_comparison_with_alignment, create_evaluation_visualizations,
    print_evaluation_summary, print_case_results
)
from src.utils.image_utils import prepare_image
from src.models.briarmbg import BriaRMBG

from huggingface_hub import snapshot_download
from accelerate.utils import set_seed

MAX_NUM_PARTS = 16

def _signed_distance_fast(mesh: trimesh.Trimesh, points: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """
    Fast SDF tailored for penetration metric:
    - First determine inside points only on candidates within expanded bbox via mesh.contains (chunked)
    - Then compute closest distances only for inside points
    - Outside points are assigned SDF=0.0 (sufficient for penetration where we only use negatives)
    """
    n = len(points)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    # Expanded bounding box to filter obvious outsides
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    diag = float(np.linalg.norm(bounds_max - bounds_min)) + 1e-12
    margin = 1e-3 * diag
    lb = bounds_min - margin
    ub = bounds_max + margin

    inside_candidates = np.all(points >= lb, axis=1) & np.all(points <= ub, axis=1)
    candidate_idx = np.nonzero(inside_candidates)[0]

    inside_mask = np.zeros(n, dtype=bool)

    # Run contains only on candidates and in chunks
    for k in range(0, len(candidate_idx), chunk):
        sl = candidate_idx[k:k+chunk]
        try:
            inside_mask[sl] = mesh.contains(points[sl])
        except Exception:
            # If contains fails, fall back to ray test per chunk
            try:
                inside_mask[sl] = mesh.ray.contains_points(points[sl])  # type: ignore
            except Exception:
                inside_mask[sl] = False

    sdf = np.zeros(n, dtype=np.float32)

    # Compute unsigned distance only for inside points
    from trimesh.proximity import closest_point  # type: ignore
    inside_idx = np.nonzero(inside_mask)[0]
    for k in range(0, len(inside_idx), chunk):
        sl = inside_idx[k:k+chunk]
        d_unsigned, _, _ = closest_point(mesh, points[sl])
        sdf[sl] = -d_unsigned.astype(np.float32)

    return sdf


def mesh_to_tsdf(mesh, size=0.3, resolution=40):
    """
    Convert mesh to TSDF using the same method as in training.
    
    Args:
        mesh: trimesh.Trimesh object
        size: Size of the workspace
        resolution: TSDF resolution
        
    Returns:
        tsdf: TSDF grid
    """
    try:
        # Create camera and simulation setup
        camera = Camera.default()
        
        # Create TSDF
        tsdf = create_tsdf(
            size=size,
            resolution=resolution,
            depth_imgs=None,  # We don't have depth images, using mesh directly
            intrinsic=camera.intrinsic,
            extrinsics=None
        )
        
        # For mesh-based TSDF, we need to convert mesh to depth images
        # This is a simplified approach - in practice, you might want to render from multiple viewpoints
        
        # Alternative: use mesh voxelization
        voxel_grid = mesh.voxelized(pitch=size/resolution)
        voxel_array = voxel_grid.matrix.astype(np.float32)
        
        # Convert to TSDF format (distance field)
        # This is a simplified conversion - proper TSDF would need signed distance
        tsdf_grid = np.where(voxel_array > 0, 1.0, -1.0)
        
        return tsdf_grid
        
    except Exception as e:
        print(f"Error converting mesh to TSDF: {e}")
        return None


def penetration_sdf_surface(
    meshes: List[trimesh.Trimesh],
    samples_per_mesh: int = 4000,
    seed: int = 0,
    aggregate: str = "max",
) -> Dict[str, Any]:
    """
    Fast multi-object penetration via surface sampling + cross SDF checks.
    Returns PSR, AD_all, AD_inside and per-object detail.
    """
    assert len(meshes) >= 2, "At least two meshes are required"
    rng = np.random.default_rng(seed)

    sampled_pts: List[np.ndarray] = []
    for m in meshes:
        pts = m.sample(samples_per_mesh)
        sampled_pts.append(pts)

    num_meshes = len(meshes)
    total_points = 0
    total_inside = 0
    total_negdepth_sum = 0.0

    per_object: List[Dict[str, Any]] = []
    pairwise_inside_ratio = np.zeros((num_meshes, num_meshes), dtype=np.float32)

    bounds_min = np.vstack([m.bounds[0] for m in meshes]).min(0)
    bounds_max = np.vstack([m.bounds[1] for m in meshes]).max(0)
    scene_diag = float(np.linalg.norm(bounds_max - bounds_min)) + 1e-12

    for i in range(num_meshes):
        pts_i = sampled_pts[i]
        n_points = len(pts_i)
        inside_any = np.zeros(n_points, dtype=bool)
        neg_depth_acc = np.zeros(n_points, dtype=np.float32)

        for j in range(num_meshes):
            if i == j:
                continue
            d_ij = _signed_distance_fast(meshes[j], pts_i)
            mask_in = d_ij < 0
            inside_any |= mask_in
            if aggregate == "max":
                neg_depth_acc[mask_in] = np.maximum(neg_depth_acc[mask_in], (-d_ij[mask_in]).astype(np.float32))
            else:
                neg_depth_acc[mask_in] += (-d_ij[mask_in]).astype(np.float32)

            pairwise_inside_ratio[i, j] = float(mask_in.mean())

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

    psr = total_inside / max(total_points, 1)
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
    }

import numpy as np
import trimesh
from trimesh.proximity import ProximityQuery
from typing import List, Dict, Any, Tuple

def _build_pq_list(meshes: List[trimesh.Trimesh]):
    # 预构建并复用 ProximityQuery（关键）
    return [ProximityQuery(m) for m in meshes]

def _aabb_overlap(a_min, a_max, b_min, b_max, pad=0.0):
    # 简单 AABB 交叠测试（带 padding）
    return not (
        ((a_max + pad) < b_min).any() or
        ((a_min - pad) > b_max).any()
    )

def penetration_sdf_surface_fast(
    meshes: List[trimesh.Trimesh],
    samples_per_mesh: int = 4000,
    seed: int = 0,
    aggregate: str = "max",
    aabb_pad: float = 0.0,      # AABB 裁剪的 padding（按需要可设成 1~5mm 或比例）
    dtype = np.float32
) -> Dict[str, Any]:
    """
    Fast multi-object penetration via surface sampling + cross SDF checks.
    Speed-ups: reuse ProximityQuery, AABB broad-phase culling, vectorized updates.
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
            # (-d_ij).clip(min=0) 等价于 max(-d_ij, 0)
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

    psr = total_inside / max(total_points, 1)
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


import os
import numpy as np
import trimesh
from typing import Dict, Any, List, Tuple

def load_meshes_from_glb(glb_path: str, min_z_extent_ratio: float = 1e-3,
                         min_volume_ratio: float = 1e-6, verbose: bool = False) -> List[trimesh.Trimesh]:
    """
    读取 GLB 中的所有 Trimesh，并把每个 child 的世界变换应用到几何上。
    过滤掉极薄/极小的伪几何（常见于导出时的碎片）。
    """
    scene = trimesh.load(glb_path, process=False)
    if not isinstance(scene, trimesh.Scene):
        # 单一网格也兼容
        return [scene]

    meshes: List[trimesh.Trimesh] = []
    # 统计用于筛选的全局尺度
    all_bounds = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            all_bounds.append(geom.bounds)
    if len(all_bounds) == 0:
        return []

    global_min = np.vstack([b[0] for b in all_bounds]).min(0)
    global_max = np.vstack([b[1] for b in all_bounds]).max(0)
    global_extent = (global_max - global_min).max()
    global_volume = max((global_max - global_min).prod(), 1e-12)

    for node_name, geom_name in scene.graph.nodes_geometry.items():
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        T = scene.graph.get(node_name)  # 节点到世界坐标的变换
        if T is not None:
            m.apply_transform(T)

        # 过滤非常小的伪网格
        extent = (m.bounds[1] - m.bounds[0])
        if extent.max() < min_z_extent_ratio * global_extent:
            if verbose:
                print(f"[skip small extent] {geom_name}, extent={extent}")
            continue
        if m.volume is not None and np.isfinite(m.volume):
            if m.volume < min_volume_ratio * global_volume:
                if verbose:
                    print(f"[skip small volume] {geom_name}, volume={m.volume}")
                continue

        # 尽量修复为封闭网格（SDF 更稳）
        if not m.is_watertight:
            m = m.fill_holes() or m
        meshes.append(m)

    return meshes


def load_meshes_from_glb(glb_path: str, min_z_extent_ratio: float = 1e-3,
                         min_volume_ratio: float = 1e-6, verbose: bool = False) -> List[trimesh.Trimesh]:
    """
    读取 GLB 中的所有 Trimesh，并把每个 child 的世界变换应用到几何上。
    过滤掉极薄/极小的伪几何（常见于导出时的碎片）。
    """
    scene = trimesh.load(glb_path, process=False)
    if not isinstance(scene, trimesh.Scene):
        # 单一网格也兼容
        return [scene]

    meshes: List[trimesh.Trimesh] = []
    # 统计用于筛选的全局尺度
    all_bounds = []
    for name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.Trimesh):
            all_bounds.append(geom.bounds)
    if len(all_bounds) == 0:
        return []

    global_min = np.vstack([b[0] for b in all_bounds]).min(0)
    global_max = np.vstack([b[1] for b in all_bounds]).max(0)
    global_extent = (global_max - global_min).max()
    global_volume = max((global_max - global_min).prod(), 1e-12)

    for node_name, geom_name in scene.graph.nodes_geometry.items():
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        T = scene.graph.get(node_name)  # 节点到世界坐标的变换
        if T is not None:
            m.apply_transform(T)

        # 过滤非常小的伪网格
        extent = (m.bounds[1] - m.bounds[0])
        if extent.max() < min_z_extent_ratio * global_extent:
            if verbose:
                print(f"[skip small extent] {geom_name}, extent={extent}")
            continue
        if m.volume is not None and np.isfinite(m.volume):
            if m.volume < min_volume_ratio * global_volume:
                if verbose:
                    print(f"[skip small volume] {geom_name}, volume={m.volume}")
                continue

        # 尽量修复为封闭网格（SDF 更稳）
        if not m.is_watertight:
            m = m.fill_holes() or m
        meshes.append(m)

    return meshes

def penetration_psr_from_meshes(meshes: List[trimesh.Trimesh],
                                samples_per_mesh: int = 4000,
                                seed: int = 0,
                                return_detail: bool = True,
                                aabb_margin_ratio: float = 0.01,
                                chunk: int = 20000) -> Dict[str, Any]:
    """
    计算场景级 PSR 与细节：
      - PSR: 所有物体表面采样点中，落在任意其它物体内部的比例
      - AD_inside: inside 点的平均负 SDF（渗透深度，越大越严重）
      - per_object: 每个物体的 inside 比率与深度
      - pairwise_inside_ratio: 两两 inside 比率矩阵（基于 i 的点落到 j 内部的比例）
    说明：
      - 使用 trimesh.proximity.signed_distance（需要 rtree，更快更稳）
      - 用 AABB 粗筛避免不相交的 pair
      - 对非 watertight 网格 SDF 可能不稳，已尽量修复，但仍建议保证 watertight
    """
    assert len(meshes) >= 2, "需要至少两个 mesh"

    rng = np.random.default_rng(seed)
    num = len(meshes)

    # 预采样表面点
    surface_points: List[np.ndarray] = []
    for m in meshes:
        # trimesh.Trimesh.sample 是按面积权重的均匀表面采样
        pts = m.sample(samples_per_mesh)
        surface_points.append(pts)

    # 预计算 AABB（加小 margin，加快跳过不可能相交的 pair）
    bounds = np.array([[m.bounds[0], m.bounds[1]] for m in meshes])  # (N,2,3)
    extents = bounds[:, 1] - bounds[:, 0]
    margins = (extents.max(axis=1, keepdims=True) * aabb_margin_ratio)  # (N,1)
    # 为简单起见，对三个轴都用同一 margin
    inflated = np.stack([bounds[:, 0] - margins, bounds[:, 1] + margins], axis=1)  # (N,2,3)

    # 准备结果容器
    total_points = 0
    total_inside = 0
    neg_depth_sum = 0.0

    per_object = []
    pairwise_inside_ratio = np.zeros((num, num), dtype=np.float32)

    # 主循环：对每个 i 的点，检查是否落入其他 j
    for i in range(num):
        pts_i = surface_points[i]
        n_i = len(pts_i)
        total_points += n_i

        inside_mask_any = np.zeros(n_i, dtype=bool)
        neg_depth_collect: List[float] = []

        # 针对每个 j，必要时才做 SDF
        for j in range(num):
            if i == j:
                continue
            # 快速 AABB 粗略相交判断
            # 如果 AABB 不重叠，就跳过
            a0, a1 = inflated[i]
            b0, b1 = inflated[j]
            overlap = np.all(a0 <= b1) and np.all(b0 <= a1)
            if not overlap:
                continue

            # 进一步：只对可能落在 j 包围盒附近的点做 SDF（减少计算）
            in_box = np.all((pts_i >= b0) & (pts_i <= b1), axis=1)
            idx = np.where(in_box)[0]
            if idx.size == 0:
                continue

            # 分块计算 SDF，避免一次性内存过大
            dists = np.empty(idx.size, dtype=np.float32)
            start = 0
            while start < idx.size:
                end = min(start + chunk, idx.size)
                sub = pts_i[idx[start:end]]
                # 需要 rtree：pip install rtree；Linux 可能需要系统安装 libspatialindex
                # sd = trimesh.proximity.signed_distance(meshes[j], sub)
                from trimesh.proximity import ProximityQuery

                # 1) 预构建每个 mesh 的 ProximityQuery（放在主循环外执行一次）
                pqs = [ProximityQuery(m) for m in meshes]

                # 2) 查询时用已缓存的 PQ，而不是每次调用 trimesh.proximity.signed_distance
                sd = pqs[j].signed_distance(sub)  # 比原来快很多
                dists[start:end] = sd.astype(np.float32)
                start = end

            inside_j = dists < 0
            # 更新 pairwise i->j 的 inside 比例（以 i 的点为分母）
            pairwise_inside_ratio[i, j] = inside_j.sum() / float(n_i)

            # 累积 “任意 j 内部” 的布尔
            inside_mask_any[idx] |= inside_j

            # 记录负深度（只统计 inside 的负值绝对值）
            if inside_j.any():
                neg_depth_collect.append((-dists[inside_j]).sum())

        # 汇总 i 的统计
        n_inside_i = inside_mask_any.sum()
        total_inside += int(n_inside_i)
        if len(neg_depth_collect) > 0:
            neg_depth_sum += float(np.sum(neg_depth_collect))

        if return_detail:
            per_object.append({
                "index": i,
                "num_points": int(n_i),
                "num_inside_any": int(n_inside_i),
                "psr_object": n_inside_i / float(n_i + 1e-12),
                "avg_depth_inside": ( (np.sum(neg_depth_collect) / (n_inside_i + 1e-12)) if n_inside_i > 0 else 0.0 )
            })

    psr = total_inside / float(total_points + 1e-12)
    ad_inside = (neg_depth_sum / (total_inside + 1e-12)) if total_inside > 0 else 0.0

    out = {
        "PSR": float(psr),
        "AD_inside": float(ad_inside),
        "total_points": int(total_points),
        "total_inside": int(total_inside),
    }
    if return_detail:
        out["per_object"] = per_object
        out["pairwise_inside_ratio"] = pairwise_inside_ratio
    return out


def penetration_psr_from_glb(glb_path: str,
                             samples_per_mesh: int = 4000,
                             seed: int = 0,
                             verbose: bool = False) -> Dict[str, Any]:
    meshes = load_meshes_from_glb(glb_path, verbose=verbose)
    if len(meshes) < 2:
        raise ValueError("GLB 中没有至少两个网格。")
    return penetration_psr_from_meshes(meshes, samples_per_mesh=samples_per_mesh, seed=seed)


def load_meshes_from_glb_for_penetration(glb_path):
    """Load meshes from GLB for penetration rate calculation"""
    scene = trimesh.load(glb_path, process=False)
    if isinstance(scene, trimesh.Trimesh):
        return [scene]
    meshes = []
    for node_name, geom_name in scene.graph.nodes_geometry.items():
        geom = scene.geometry[geom_name]
        if not isinstance(geom, trimesh.Trimesh):
            continue
        m = geom.copy()
        T = scene.graph.get(node_name)
        if T is not None:
            m.apply_transform(T)
        # 可选：轻量修复以提高 contains 的稳定性
        m.remove_degenerate_faces()
        m.remove_unreferenced_vertices()
        m.fill_holes()  # 可能改变几何，按需使用
        meshes.append(m)
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


def compute_penetration_rate(glb_path, resolution=128):
    """
    Compute penetration rate for a GLB file
    
    Args:
        glb_path: Path to GLB file
        resolution: Grid resolution for sampling
        
    Returns:
        Dictionary with penetration metrics
    """
    meshes = load_meshes_from_glb_for_penetration(glb_path)
    assert len(meshes) > 0, "No meshes found."
    bounds = np.array([m.bounds for m in meshes])  # [M, 2, 3]
    lo = bounds[:,0,:].min(axis=0)
    hi = bounds[:,1,:].max(axis=0)
    points, shape = make_grid(lo, hi, resolution=resolution)
    PR, U, S, pair_mat = penetration_rate_from_points(meshes, points)
    return {
        "penetration_rate": PR,                 # 0~1，越大穿插越严重
        "union_points": int(U),
        "sum_individual_points": int(S),
        "resolution": resolution,
        "pairwise_overlap_matrix": pair_mat     # 可选，用于定位问题
    }


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


class PartCrafterEvaluator:
    def __init__(self, 
                 model_path: str = "pretrained_weights/PartCrafter-Scene",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 build_gaps: bool = True,
                 load_trained_checkpoint: str = None,
                 checkpoint_iteration: int = None):
        """
        Initialize PartCrafter evaluator
        
        Args:
            model_path: Model weights path (base model for VAE, feature extractor, etc.)
            device: Computing device
            dtype: Data type
            build_gaps: Whether to build GAPS tools for evaluation
            load_trained_checkpoint: Path to trained checkpoint directory (e.g., "runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512")
            checkpoint_iteration: Checkpoint iteration number (e.g., 17000)
        """
        self.device = device
        self.dtype = dtype
        self.build_gaps = build_gaps
        self.load_trained_checkpoint = load_trained_checkpoint
        self.checkpoint_iteration = checkpoint_iteration
        
        # Download and load base model components - exactly match inference_partcrafter.py
        print(f"Downloading base model weights to: {model_path}")
        snapshot_download(repo_id="wgsxm/PartCrafter", local_dir=model_path)
        
        # Download RMBG weights - exactly match inference_partcrafter.py
        rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
        snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)
        
        # init rmbg model for background removal - exactly match inference_partcrafter.py
        self.rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
        self.rmbg_net.eval()
        
        print("Loading PartCrafter model...")
        
        if load_trained_checkpoint is not None and checkpoint_iteration is not None:
            # Load trained checkpoint - following train_partcrafter.py pattern
            print(f"Loading trained checkpoint from: {load_trained_checkpoint}")
            print(f"Checkpoint iteration: {checkpoint_iteration}")
            
            # Load base model components from pretrained weights
            vae = TripoSGVAEModel.from_pretrained(model_path, subfolder="vae")
            feature_extractor_dinov2 = BitImageProcessor.from_pretrained(model_path, subfolder="feature_extractor_dinov2")
            image_encoder_dinov2 = Dinov2Model.from_pretrained(model_path, subfolder="image_encoder_dinov2")
            noise_scheduler = RectifiedFlowScheduler.from_pretrained(model_path, subfolder="scheduler")
            
            # Load trained transformer from checkpoint
            checkpoint_path = os.path.join(load_trained_checkpoint, "checkpoints", f"{checkpoint_iteration:06d}")
            print(f"Loading transformer from: {checkpoint_path}")
            
            # Load transformer with EMA weights (following train_partcrafter.py pattern)
            transformer, loading_info = PartCrafterDiTModel.from_pretrained(
                checkpoint_path, 
                subfolder="transformer_ema",
                low_cpu_mem_usage=False, 
                output_loading_info=True
            )
            
            # Create pipeline with trained components
            self.pipeline = PartCrafterPipeline(
                vae=vae,
                transformer=transformer,
                scheduler=noise_scheduler,
                feature_extractor_dinov2=feature_extractor_dinov2,
                image_encoder_dinov2=image_encoder_dinov2,
            ).to(device, dtype)
            
            print(f"Loaded trained checkpoint successfully!")
            if loading_info:
                print(f"Loading info: {loading_info}")
        else:
            # Load standard pretrained model
            self.pipeline = PartCrafterPipeline.from_pretrained(model_path).to(device, dtype)
            print("Loaded standard pretrained model")
        
        print("Model loading completed!")
        
        # Set seed for reproducibility - same as inference_partcrafter.py
        set_seed(0)
        
        # Setup GAPS if requested
        if self.build_gaps:
            setup_gaps_tools()
    
    def save_metrics_to_txt(self, case_name: str, metrics: Dict[str, Any], output_path: str):
        """
        Save metrics to a readable text file
        
        Args:
            case_name: Name of the test case
            metrics: Dictionary containing evaluation metrics
            output_path: Path to save the metrics.txt file
        """
        with open(output_path, 'w') as f:
            f.write(f"PartCrafter Evaluation Results\n")
            f.write(f"Case: {case_name}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write(f"  Chamfer Distance: {metrics.get('chamfer_distance', 'N/A'):.6f}\n")
            f.write(f"  F-Score: {metrics.get('f_score', 'N/A'):.6f}\n")
            f.write(f"  IoU: {metrics.get('iou', 'N/A'):.6f}\n")
            f.write(f"  Scene IoU: {metrics.get('scene_iou', 'N/A'):.6f}\n")
            f.write("\n")
            
            # Per-object metrics
            if 'per_object_cds' in metrics and metrics['per_object_cds']:
                f.write("PER-OBJECT METRICS:\n")
                f.write(f"  Number of GT objects: {len(metrics['per_object_cds'])}\n")
                f.write(f"  Number of predicted objects: {len(metrics.get('per_object_fscores', []))}\n")
                f.write("\n")
                
                # Individual object metrics
                for i, (cd, fscore, iou) in enumerate(zip(
                    metrics['per_object_cds'],
                    metrics['per_object_fscores'],
                    metrics['per_object_ious']
                )):
                    f.write(f"  Object {i+1}:\n")
                    f.write(f"    CD: {cd:.6f}\n")
                    f.write(f"    F-Score: {fscore:.6f}\n")
                    f.write(f"    IoU: {iou:.6f}\n")
                    f.write("\n")
            
            # Alignment info
            if 'aligned_gt_scene_path' in metrics:
                f.write("ALIGNMENT INFO:\n")
                f.write(f"  GT Scene: {metrics['aligned_gt_scene_path']}\n")
                f.write(f"  Predicted Scene: {metrics['aligned_pred_merged_path']}\n")
                f.write("\n")
            
            # Additional info
            f.write("ADDITIONAL INFO:\n")
            f.write(f"  Evaluation completed successfully\n")
            f.write(f"  All meshes aligned using GAPS sim(3) registration\n")
            f.write(f"  Metrics computed on aligned meshes\n")
    
    def save_summary_metrics_to_txt(self, results: List[Dict], output_dir: str):
        """
        Save summary metrics to a text file in the main output directory
        
        Args:
            results: List of evaluation results
            output_dir: Main output directory
        """
        summary_path = os.path.join(output_dir, "summary_metrics.txt")
        
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        
        with open(summary_path, 'w') as f:
            f.write(f"PartCrafter Evaluation Summary\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total cases: {len(results)}\n")
            f.write(f"  Successful cases: {len(successful_results)}\n")
            f.write(f"  Failed cases: {len(failed_results)}\n")
            f.write(f"  Success rate: {len(successful_results)/len(results)*100:.1f}%\n")
            f.write("\n")
            
            if successful_results:
                # Calculate summary metrics
                all_cds = [r['metrics']['chamfer_distance'] for r in successful_results]
                all_fscores = [r['metrics']['f_score'] for r in successful_results]
                all_ious = [r['metrics']['iou'] for r in successful_results]
                all_scene_ious = [r['metrics']['scene_iou'] for r in successful_results]
                
                f.write("SUMMARY METRICS:\n")
                f.write(f"  Chamfer Distance:\n")
                f.write(f"    Mean: {np.mean(all_cds):.6f}\n")
                f.write(f"    Std: {np.std(all_cds):.6f}\n")
                f.write(f"    Min: {np.min(all_cds):.6f}\n")
                f.write(f"    Max: {np.max(all_cds):.6f}\n")
                f.write("\n")
                
                f.write(f"  F-Score:\n")
                f.write(f"    Mean: {np.mean(all_fscores):.6f}\n")
                f.write(f"    Std: {np.std(all_fscores):.6f}\n")
                f.write(f"    Min: {np.min(all_fscores):.6f}\n")
                f.write(f"    Max: {np.max(all_fscores):.6f}\n")
                f.write("\n")
                
                f.write(f"  IoU:\n")
                f.write(f"    Mean: {np.mean(all_ious):.6f}\n")
                f.write(f"    Std: {np.std(all_ious):.6f}\n")
                f.write(f"    Min: {np.min(all_ious):.6f}\n")
                f.write(f"    Max: {np.max(all_ious):.6f}\n")
                f.write("\n")
                
                f.write(f"  Scene IoU:\n")
                f.write(f"    Mean: {np.mean(all_scene_ious):.6f}\n")
                f.write(f"    Std: {np.std(all_scene_ious):.6f}\n")
                f.write(f"    Min: {np.min(all_scene_ious):.6f}\n")
                f.write(f"    Max: {np.max(all_scene_ious):.6f}\n")
                f.write("\n")
            
            # Failed cases
            if failed_results:
                f.write("FAILED CASES:\n")
                for result in failed_results:
                    f.write(f"  {result['case_name']}: {result.get('error', 'Unknown error')}\n")
                f.write("\n")
            
            f.write("EVALUATION COMPLETED\n")
            f.write(f"Results saved to: {output_dir}\n")
        
        print(f"Summary metrics saved to: {summary_path}")
        return summary_path
    

    

    
    def load_test_config(self, config_path: str) -> List[Dict]:
        """Load test configuration"""
        with open(config_path, 'r') as f:
            configs = json.load(f)
        return configs
    
    def load_gt_mesh(self, mesh_path: str) -> trimesh.Scene:
        """Load Ground Truth mesh"""
        mesh = trimesh.load(mesh_path, process=False)
        return mesh
    
# from typing import Tuple, List
# import os
# import trimesh

    def check_existing_results(
        self, case_name: str, output_dir: str
    ) -> Tuple[bool, str, List[trimesh.Trimesh], List[trimesh.Trimesh]]:
        """
        Returns: (exists, mesh_dir, pred_meshes, gt_meshes)
        """

        def _as_trimesh_list(obj) -> List[trimesh.Trimesh]:
            if obj is None:
                return []
            if isinstance(obj, trimesh.Scene):
                return list(obj.geometry.values())
            if isinstance(obj, trimesh.Trimesh):
                return [obj]
            return []

        def _load_parts(dir_path: str, prefix: str) -> List[trimesh.Trimesh]:
            meshes: List[trimesh.Trimesh] = []
            i = 0
            while True:
                p = os.path.join(dir_path, f"{prefix}_{i:02d}.glb")
                if not os.path.exists(p):
                    break
                m = trimesh.load(p, process=False)
                meshes.extend(_as_trimesh_list(m))
                i += 1
            return meshes

        mesh_dir = os.path.join(output_dir, case_name)
        if not os.path.exists(mesh_dir):
            return False, mesh_dir, [], []

        pred_meshes = _load_parts(mesh_dir, "pred_part")
        gt_meshes   = _load_parts(mesh_dir, "gt_part")

        if not pred_meshes:
            pred_merged_path = os.path.join(mesh_dir, "pred_merged.glb")
            if os.path.exists(pred_merged_path):
                pred_meshes = _as_trimesh_list(trimesh.load(pred_merged_path, process=False))

        if not gt_meshes:
            gt_merged_path = os.path.join(mesh_dir, "gt_merged.glb")
            if os.path.exists(gt_merged_path):
                gt_meshes = _as_trimesh_list(trimesh.load(gt_merged_path, process=False))

        exists = bool(pred_meshes or gt_meshes)
        return exists, mesh_dir, pred_meshes, gt_meshes

    
    @torch.no_grad()
    def run_inference(self, 
                     image_path: str, 
                     num_parts: int,
                     seed: int = 0,
                     num_tokens: int = 1024,
                     num_inference_steps: int = 50,
                     guidance_scale: float = 7.0,
                     max_num_expanded_coords: int = 1e9,
                     use_flash_decoder: bool = False,
                     rmbg: bool = False,
                     rmbg_net: Any = None,
                     dtype: torch.dtype = torch.float16,
                     device: str = "cuda") -> Tuple[List[trimesh.Trimesh], Image.Image]:
        """
        Run PartCrafter inference - exactly matching inference_partcrafter.py run_triposg function
        
        Returns:
            generated_meshes: List of generated meshes
            processed_image: Processed input image
        """
        # Exactly match inference_partcrafter.py run_triposg function
        assert 1 <= num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
        if rmbg:
            img_pil = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)
        else:
            img_pil = Image.open(image_path)
        start_time = time.time()
        outputs = self.pipeline(
            image=[img_pil] * num_parts,
            attention_kwargs={"num_parts": num_parts},
            num_tokens=num_tokens,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_num_expanded_coords=max_num_expanded_coords,
            use_flash_decoder=use_flash_decoder,
        ).meshes
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time:.2f} seconds")
        for i in range(len(outputs)):
            if outputs[i] is None:
                # If the generated mesh is None (decoding error), use a dummy mesh
                outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
        return outputs, img_pil
    

    

    

    

    
    def evaluate_single_case(self, 
                           config: Dict,
                           output_dir: str,
                           num_samples: int = 10000,
                           use_existing_results: bool = True,
                           force_inference: bool = False,
                           inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate a single test case
        
        Returns:
            Dictionary containing evaluation results
        """
        case_name = Path(config['mesh_path']).stem
        print(f"\nEvaluating case: {case_name}")
        
        try:
            # Check if existing results are available
            has_existing, mesh_dir, pred_meshes, gt_meshes = self.check_existing_results(case_name, output_dir)
            
            if has_existing and use_existing_results and not force_inference:
                # Use existing results - skip inference but still need GT mesh for metrics
                # Load GT mesh from original path for metrics computation
                gt_mesh = self.load_gt_mesh(config['mesh_path'])
                print(f"Loaded {len(pred_meshes)} predicted meshes and {len(gt_meshes)} GT meshes from existing results")
            else:
                # Load GT mesh and run inference
                if has_existing and force_inference:
                    print(f"Force running inference for case: {case_name} (overwriting existing results)")
                else:
                    print(f"Running inference for case: {case_name}")
                
                gt_mesh = self.load_gt_mesh(config['mesh_path'])
                
                # Create output directory for this case
                case_output_dir = os.path.join(output_dir, case_name)
                os.makedirs(case_output_dir, exist_ok=True)
                
                # Run PartCrafter inference with exact same parameters as inference_partcrafter.py
                pred_meshes, input_image = self.run_inference(
                    image_path=config['image_path'],
                    num_parts=config['num_parts'],
                    seed=inference_args.get('seed', 0) if inference_args else 0,
                    num_tokens=inference_args.get('num_tokens', 1024) if inference_args else 1024,
                    num_inference_steps=inference_args.get('num_inference_steps', 50) if inference_args else 50,
                    guidance_scale=inference_args.get('guidance_scale', 7.0) if inference_args else 7.0,
                    max_num_expanded_coords=inference_args.get('max_num_expanded_coords', int(1e9)) if inference_args else int(1e9),
                    use_flash_decoder=inference_args.get('use_flash_decoder', False) if inference_args else False,
                    rmbg=inference_args.get('rmbg', False) if inference_args else False,
                    rmbg_net=self.rmbg_net,
                    dtype=self.dtype,
                    device=self.device
                )
                # pred_meshes[0].export(os.path.join(case_output_dir, "pred_part_00.glb"))
            
            # Always compute metrics with alignment (whether using existing results or not)
            print(f"Computing metrics with alignment for {len(pred_meshes)} predicted meshes...")
            
            # Extract GT meshes from gt_mesh if not already loaded
            if not gt_meshes:
                if isinstance(gt_mesh, trimesh.Scene):
                    gt_meshes = list(gt_mesh.geometry.values())
                elif isinstance(gt_mesh, trimesh.Trimesh):
                    gt_meshes = [gt_mesh]
                else:
                    gt_meshes = []
            
            # Create output directory for this case (if not already created)
            case_output_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_output_dir, exist_ok=True)
            
            # Save meshes for alignment
            pred_merged_path = os.path.join(case_output_dir, "pred_merged.ply")
            gt_merged_path = os.path.join(case_output_dir, "gt_merged.ply")
            aligned_pred_path = os.path.join(case_output_dir, "aligned_pred_merged.ply")
            
            # Merge and save meshes
            assert pred_meshes is not None
            assert gt_meshes is not None
            pred_merged = trimesh.util.concatenate(pred_meshes)
            pred_merged.export(pred_merged_path)
       
            gt_merged = trimesh.util.concatenate(gt_meshes)
            gt_merged.export(gt_merged_path)
    
            # Run GAPS alignment if both meshes exist
            print("Running GAPS alignment with transformation extraction...")
                
            # Get aligned merged mesh, transformation matrix, and scale factor
            gt_merged_aligned, aligned_pred_merged, transformation_matrix, scale_factor = align_merged_meshes_with_gaps_and_get_transform(gt_merged, pred_merged)
                    
            aligned_pred_merged.export(os.path.join(case_output_dir, "aligned_pred_merged.glb"))
            gt_merged.export(os.path.join(case_output_dir, "gt_merged.glb"))

            print(f"GAPS alignment successful")
            print(f"Transformation matrix shape: {transformation_matrix.shape}")
            print(f"Scale factor: {scale_factor}")
            print(f"Transformation matrix:\n{transformation_matrix}")

            assert np.allclose(transformation_matrix, np.eye(4)) != True
            assert scale_factor != 1.0 
            
            # Apply the transformation to individual predicted meshes
            print(f"Applying transformation to {len(pred_meshes)} individual predicted meshes...")
            aligned_pred_meshes = apply_gaps_transformation_to_meshes(pred_meshes, transformation_matrix)
            print(f"Successfully applied transformation to {len(aligned_pred_meshes)} individual predicted meshes")
            
            # Save transformation info for debugging
            transformation_info = {
                'transformation_matrix': transformation_matrix.tolist(),
                'scale_factor': scale_factor,
                'matrix_shape': transformation_matrix.shape,
                'is_identity': bool(np.allclose(transformation_matrix, np.eye(4)))
            }
                    
            # Save transformation info to file
            transformation_path = os.path.join(case_output_dir, "transformation_info.json")
            with open(transformation_path, 'w') as f:
                json.dump(transformation_info, f, indent=2)
            print(f"Transformation info saved to: {transformation_path}")
       
            aligned_gt_scene = trimesh.Scene(gt_meshes)
            aligned_pred_scene = trimesh.Scene(aligned_pred_meshes)
            
            # Temporarily comment out metrics computation
            # print(f"Computing metrics with {len(gt_meshes)} GT meshes and {len(aligned_pred_meshes)} aligned predicted meshes")
            # metrics = compute_aligned_metrics(gt_meshes, aligned_pred_meshes, num_samples)
            
            # Create dummy metrics for now
            metrics = {
                'chamfer_distance': 0.0,
                'f_score': 0.0,
                'iou': 0.0,
                'scene_iou': 0.0,
                'per_object_cds': [],
                'per_object_fscores': [],
                'per_object_ious': []
            }

            # Save aligned predicted meshes
            if aligned_pred_meshes and len(aligned_pred_meshes) > 0:
                # Save each aligned predicted mesh as aligned_pred_obj_*.ply
                print(f"Saving {len(aligned_pred_meshes)} aligned predicted meshes as PLY files...")
                for i, mesh in enumerate(aligned_pred_meshes):
                    ply_filename = f"aligned_pred_obj_{i}.ply"
                    ply_path = os.path.join(case_output_dir, ply_filename)
                    mesh.export(ply_path)
                    print(f"  Saved: {ply_filename}")
                
            # Save GT meshes
            if gt_meshes and len(gt_meshes) > 0:
                # Save each GT mesh as gt_object_*.ply
                print(f"Saving {len(gt_meshes)} GT meshes as PLY files...")
                for i, mesh in enumerate(gt_meshes):
                    ply_filename = f"gt_object_{i}.ply"
                    ply_path = os.path.join(case_output_dir, ply_filename)
                    mesh.export(ply_path)
                    print(f"  Saved: {ply_filename}")
            
            # Add aligned scene file paths (avoid storing Scene objects which are not JSON serializable)
            metrics['aligned_gt_scene_path'] = os.path.join(case_output_dir, "gt_merged.glb")
            metrics['aligned_pred_merged_path'] = os.path.join(case_output_dir, "aligned_pred_merged.glb")

            # Compute multi-object penetration score on aligned predicted meshes (SDF-based)
            # TEMPORARILY COMMENTED OUT - will be re-enabled later
            """
            try:
                # Method 1: SDF-based penetration (surface sampling)
                penetration_res = penetration_psr_from_meshes(
                    aligned_pred_meshes,
                    samples_per_mesh=4000,
                    seed=42,
                    return_detail=True,
                    aabb_margin_ratio=0.01,
                    chunk=20000
                )
                # Ensure JSON serializable (e.g., numpy arrays to lists)
                if isinstance(penetration_res.get("pairwise_inside_ratio"), np.ndarray):
                    penetration_res["pairwise_inside_ratio"] = penetration_res["pairwise_inside_ratio"].tolist()
                
                # Method 2: Grid-based penetration rate (volume sampling)
                try:
                    print(f"Computing grid-based penetration rate for {len(aligned_pred_meshes)} aligned predicted meshes...")
                    
                    # 根据mesh数量动态调整分辨率
                    if len(aligned_pred_meshes) <= 5:
                        resolution = 64  # 少量mesh用高分辨率
                    elif len(aligned_pred_meshes) <= 10:
                        resolution = 48  # 中等数量用中等分辨率
                    else:
                        resolution = 32  # 大量mesh用低分辨率
                    
                    print(f"Using resolution {resolution} for {len(aligned_pred_meshes)} meshes")
                    
                    penetration_rate_res = compute_penetration_rate_from_meshes(
                        aligned_pred_meshes, 
                        resolution=resolution
                    )
                    
                    # Ensure JSON serializable
                    if isinstance(penetration_rate_res.get("pairwise_overlap_matrix"), np.ndarray):
                        penetration_rate_res["pairwise_overlap_matrix"] = penetration_rate_res["pairwise_overlap_matrix"].tolist()
                    
                    # Combine both methods
                    penetration_res["grid_based"] = penetration_rate_res
                    print(f"Grid-based penetration rate: {penetration_rate_res['penetration_rate']:.4f}")
                    print(f"Valid meshes: {penetration_rate_res.get('valid_mesh_count', 'N/A')}/{penetration_rate_res.get('total_mesh_count', 'N/A')}")
                    
                except Exception as e2:
                    print(f"[WARN] Grid-based penetration rate failed: {e2}")
                    import traceback
                    traceback.print_exc()
                
                metrics['penetration'] = penetration_res
                print(f"SDF-based PSR: {penetration_res.get('PSR', 'N/A'):.4f}")
            except Exception as e:
                print(f"[WARN] Penetration metric failed: {e}")
            """
            print("[INFO] Penetration metrics temporarily disabled")
            
            comparison_path = None
            
            # Save alignment results if not using existing results or if alignment results don't exist
            # if not has_existing or not os.path.exists(os.path.join(mesh_dir, "gt_merged_aligned.glb")):
            #     mesh_dir = save_meshes_with_alignment(
            #         pred_meshes, gt_mesh, output_dir, case_name,
            #         aligned_gt_scene, aligned_pred_scene, metrics
            #     )
            
            result = {
                'case_name': case_name,
                'num_parts': config['num_parts'],
                'metrics': metrics,
                'comparison_image': comparison_path,
                'mesh_dir': mesh_dir,
                'success': True
            }
            
            print_case_results(case_name, metrics)
            
            # Save metrics to a text file in the scene folder
            metrics_txt_path = os.path.join(case_output_dir, "metrics.txt")
            self.save_metrics_to_txt(case_name, metrics, metrics_txt_path)
            print(f"Metrics saved to: {metrics_txt_path}")
            
            return result
            
        except Exception as e:
            print(f"Error evaluating case {case_name}: {e}")
            return {
                'case_name': case_name,
                'num_parts': config.get('num_parts', 0),
                'metrics': {
                    'chamfer_distance': float('inf'),
                    'f_score': 0.0,
                    'iou': 0.0,
                    'scene_iou': 0.0
                },

                'error': str(e),
                'success': False
            }
    
    def evaluate_dataset(self, 
                        config_path: str,
                        output_dir: str,
                        num_samples: int = 10000,
                        use_existing_results: bool = True,
                        force_inference: bool = False,
                        inference_args: Dict = None) -> Dict[str, Any]:
        """
        Evaluate the entire dataset
        
        Returns:
            Dictionary containing all evaluation results
        """
        print(f"Starting dataset evaluation: {config_path}")
        
        # Load test configuration
        configs = self.load_test_config(config_path)
        print(f"Found {len(configs)} test cases")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Evaluate each case
        results = []
        for config in tqdm(configs, desc="Evaluation progress"):
            result = self.evaluate_single_case(
                config, output_dir, num_samples, 
                use_existing_results, force_inference, inference_args
            )
            results.append(result)
        
        # Calculate overall statistics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            metrics_summary = {
                'chamfer_distance': {
                    'mean': np.mean([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'std': np.std([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'min': np.min([r['metrics']['chamfer_distance'] for r in successful_results]),
                    'max': np.max([r['metrics']['chamfer_distance'] for r in successful_results])
                },
                'f_score': {
                    'mean': np.mean([r['metrics']['f_score'] for r in successful_results]),
                    'std': np.std([r['metrics']['f_score'] for r in successful_results]),
                    'min': np.min([r['metrics']['f_score'] for r in successful_results]),
                    'max': np.max([r['metrics']['f_score'] for r in successful_results])
                },
                'iou': {
                    'mean': np.mean([r['metrics']['iou'] for r in successful_results]),
                    'std': np.std([r['metrics']['iou'] for r in successful_results]),
                    'min': np.min([r['metrics']['iou'] for r in successful_results]),
                    'max': np.max([r['metrics']['iou'] for r in successful_results])
                },
                'scene_iou': {
                    'mean': np.mean([r['metrics']['scene_iou'] for r in successful_results]),
                    'std': np.std([r['metrics']['scene_iou'] for r in successful_results]),
                    'min': np.min([r['metrics']['scene_iou'] for r in successful_results]),
                    'max': np.max([r['metrics']['scene_iou'] for r in successful_results])
                },
                'per_object_metrics': {
                    'all_cds': [cd for r in successful_results for cd in r['metrics'].get('per_object_cds', [])],
                    'all_fscores': [fscore for r in successful_results for fscore in r['metrics'].get('per_object_fscores', [])],
                    'all_ious': [iou for r in successful_results for iou in r['metrics'].get('per_object_ious', [])]
                }
            }
        else:
            metrics_summary = {}
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'config_path': config_path,
                'total_cases': len(configs),
                'successful_cases': len(successful_results),
                'failed_cases': len(results) - len(successful_results),
                'metrics_summary': metrics_summary,
                'detailed_results': results
            }, f, indent=2)
        
        # Create results table
        if successful_results:
            df_data = []
            for result in successful_results:
                df_data.append({
                    'case_name': result['case_name'],
                    'num_parts': result['num_parts'],
                    'chamfer_distance': result['metrics']['chamfer_distance'],
                    'chamfer_distance_std': result['metrics'].get('chamfer_distance_std', 0.0),
                    'f_score': result['metrics']['f_score'],
                    'f_score_std': result['metrics'].get('f_score_std', 0.0),
                    'iou': result['metrics']['iou'],
                    'iou_std': result['metrics'].get('iou_std', 0.0),
                    'scene_iou': result['metrics']['scene_iou'],
                    'num_gt_objects': len(result['metrics'].get('per_object_cds', [])),
                    'num_pred_objects': result['num_parts']
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(os.path.join(output_dir, 'evaluation_results.csv'), index=False)
            
            # Create visualization charts
            create_evaluation_visualizations(df, output_dir)
        
        # Print summary
        print_evaluation_summary(metrics_summary, len(configs), len(successful_results))
        
        # Save summary metrics to text file
        self.save_summary_metrics_to_txt(results, output_dir)
        
        print(f"\nResults saved to: {output_dir}")
        
        return {
            'results': results,
            'metrics_summary': metrics_summary,
            'output_dir': output_dir
        }
    


def save_meshes_standalone(pred_meshes, gt_mesh, output_path, case_name="test_case"):
    """
    Standalone function to save meshes as GLB files
    
    Args:
        pred_meshes: List of predicted trimesh.Trimesh objects
        gt_mesh: Ground Truth trimesh.Scene or trimesh.Trimesh object
        output_path: Directory to save the GLB files
        case_name: Name for the case (used in file naming)
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save individual predicted meshes
    for i, mesh in enumerate(pred_meshes):
        mesh.export(os.path.join(output_path, f"pred_part_{i:02d}.glb"))
    
    # Save merged predicted mesh
    if pred_meshes:
        merged_mesh = trimesh.util.concatenate(pred_meshes)
        merged_mesh.export(os.path.join(output_path, "pred_merged.glb"))
    
    # Save GT mesh
    if isinstance(gt_mesh, trimesh.Scene):
        gt_mesh.export(os.path.join(output_path, "gt_merged.glb"))
    elif isinstance(gt_mesh, trimesh.Trimesh):
        gt_mesh.export(os.path.join(output_path, "gt_merged.glb"))
    
    print(f"All meshes saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='PartCrafter large-scale evaluation script')
    parser.add_argument('--config_path', type=str, 
                    #    default='data/preprocessed_data_scenes_objects_demo_test/objects_demo_configs.json',
                        default = '/home/ran.ding/messy-kitchen/PartCrafter/data/preprocessed_data_messy_kitchen_scenes_part2/messy_kitchen_test_100.json',
                       help='Test configuration file path')
    parser.add_argument('--output_dir', type=str, 
                       default='./results/evaluation_objects_demo_test_100',
                       help='Output directory')
    parser.add_argument('--model_path', type=str,
                       default='pretrained_weights/PartCrafter',
                       help='Base model weights path (for VAE, feature extractor, etc.)')
    parser.add_argument('--load_trained_checkpoint', type=str, 
                       default='/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512',
                       help='Path to trained checkpoint directory (e.g., "runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512")')
    parser.add_argument('--checkpoint_iteration', type=int, default=17000,
                       help='Checkpoint iteration number (e.g., 17000)')
    parser.add_argument('--device', type=str, default='cuda', help='Computing device')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of evaluation sample points')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_tokens', type=int, default=1024, help='Number of tokens for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.0, help='Guidance scale for generation')
    parser.add_argument('--max_num_expanded_coords', type=int, default=1e9, help='Maximum number of expanded coordinates')
    parser.add_argument('--use_flash_decoder', action='store_true', help='Use flash decoder')
    parser.add_argument('--rmbg', action='store_true', help='Use background removal')
    parser.add_argument('--build_gaps', action='store_true', default=True,
                       help='Setup GAPS tools for evaluation (default: True, assumes GAPS is pre-installed)')
    parser.add_argument('--use_existing_results', action='store_true', default=True,
                       help='Use existing prediction results if available (default: True)')
    parser.add_argument('--force_inference', action='store_true', default=False,
                       help='Force running inference even if existing results are found')
    parser.add_argument('--max_test_cases', type=int, default=None,
                       help='Maximum number of test cases to evaluate (if None, evaluate all)')
    parser.add_argument('--test_random_seed', type=int, default=0,
                       help='Random seed for test case sampling')

    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Create evaluator
    evaluator = PartCrafterEvaluator(
        model_path=args.model_path,
        device=args.device,
        dtype=torch.float16,
        build_gaps=args.build_gaps,
        load_trained_checkpoint=args.load_trained_checkpoint,
        checkpoint_iteration=args.checkpoint_iteration
    )
    
    # Prepare inference arguments
    inference_args = {
        'seed': args.seed,
        'num_tokens': args.num_tokens,
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale,
        'max_num_expanded_coords': args.max_num_expanded_coords,
        'use_flash_decoder': args.use_flash_decoder,
        'rmbg': args.rmbg
    }
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        config_path=args.config_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_existing_results=args.use_existing_results,
        force_inference=args.force_inference,
        inference_args=inference_args
    )
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
