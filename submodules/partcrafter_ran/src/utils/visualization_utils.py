"""
Visualization utilities for PartCrafter evaluation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import trimesh

from .render_utils import render_single_view
from .data_utils import get_colored_mesh_composition


def render_comparison_with_alignment(gt_mesh: trimesh.Scene, pred_meshes: list,
                                   input_image: Image.Image, output_dir: str, case_name: str,
                                   aligned_gt_scene: trimesh.Scene = None,
                                   aligned_pred_scene: trimesh.Scene = None) -> str:
    """
    Render comparison images with optional alignment support
    
    Args:
        gt_mesh: Ground Truth mesh scene
        pred_meshes: List of predicted meshes
        input_image: Input image
        output_dir: Output directory
        case_name: Case name
        aligned_gt_scene: Aligned GT scene
        aligned_pred_scene: Aligned predicted scene
        
    Returns:
        Path to saved rendered image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use aligned meshes for rendering if available
    if aligned_gt_scene is not None and aligned_pred_scene is not None:
        # Render aligned meshes
        gt_meshes = aligned_gt_scene.dump()
        pred_meshes = aligned_pred_scene.dump()
        
        if len(gt_meshes) > 1:
            gt_merged = get_colored_mesh_composition(gt_meshes)
        else:
            gt_merged = gt_meshes[0]
            
        if len(pred_meshes) > 1:
            pred_merged = get_colored_mesh_composition(pred_meshes)
        else:
            pred_merged = pred_meshes[0]
    else:
        # Use original meshes
        if isinstance(gt_mesh, trimesh.Scene):
            gt_meshes = gt_mesh.dump()
            if len(gt_meshes) > 1:
                gt_merged = get_colored_mesh_composition(gt_meshes)
            else:
                gt_merged = gt_meshes[0]
        else:
            gt_merged = gt_mesh
        
        # Merge predicted meshes
        if len(pred_meshes) > 1:
            pred_merged = get_colored_mesh_composition(pred_meshes)
        else:
            pred_merged = pred_meshes[0]
    
    # Render GT and prediction
    gt_image = render_single_view(
        gt_merged,
        radius=4,
        image_size=(512, 512),
        light_intensity=1.5,
        num_env_lights=36,
        return_type='pil'
    )
    
    pred_image = render_single_view(
        pred_merged,
        radius=4,
        image_size=(512, 512),
        light_intensity=1.5,
        num_env_lights=36,
        return_type='pil'
    )
    
    # Create comparison image
    comparison_image = Image.new('RGB', (1536, 512))
    comparison_image.paste(input_image.resize((512, 512)), (0, 0))
    comparison_image.paste(gt_image, (512, 0))
    comparison_image.paste(pred_image, (1024, 0))
    
    # Save comparison image
    comparison_path = os.path.join(output_dir, f"{case_name}_comparison.png")
    comparison_image.save(comparison_path)
    
    return comparison_path


def create_evaluation_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualization charts for evaluation results"""
    plt.style.use('seaborn-v0_8')
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PartCrafter Multi-Object Evaluation Results', fontsize=16, fontweight='bold')
    
    # Chamfer Distance distribution
    axes[0, 0].hist(df['chamfer_distance'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Chamfer Distance Distribution')
    axes[0, 0].set_xlabel('Chamfer Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['chamfer_distance'].mean(), color='red', linestyle='--', label=f'Mean: {df["chamfer_distance"].mean():.4f}')
    axes[0, 0].legend()
    
    # F-Score distribution
    axes[0, 1].hist(df['f_score'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('F-Score Distribution')
    axes[0, 1].set_xlabel('F-Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['f_score'].mean(), color='red', linestyle='--', label=f'Mean: {df["f_score"].mean():.4f}')
    axes[0, 1].legend()
    
    # IoU distribution
    axes[0, 2].hist(df['iou'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_title('IoU Distribution')
    axes[0, 2].set_xlabel('IoU')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(df['iou'].mean(), color='red', linestyle='--', label=f'Mean: {df["iou"].mean():.4f}')
    axes[0, 2].legend()
    
    # Part count vs Chamfer Distance
    scatter1 = axes[1, 0].scatter(df['num_parts'], df['chamfer_distance'], 
                                c=df['f_score'], cmap='viridis', alpha=0.7, s=50)
    axes[1, 0].set_title('Part Count vs Chamfer Distance')
    axes[1, 0].set_xlabel('Number of Parts')
    axes[1, 0].set_ylabel('Chamfer Distance')
    plt.colorbar(scatter1, ax=axes[1, 0], label='F-Score')
    
    # GT vs Pred object count
    axes[1, 1].scatter(df['num_gt_objects'], df['num_pred_objects'], alpha=0.7, s=50, color='purple')
    axes[1, 1].plot([0, max(df['num_gt_objects'].max(), df['num_pred_objects'].max())], 
                   [0, max(df['num_gt_objects'].max(), df['num_pred_objects'].max())], 
                   'r--', alpha=0.5)
    axes[1, 1].set_title('GT vs Predicted Object Count')
    axes[1, 1].set_xlabel('GT Objects')
    axes[1, 1].set_ylabel('Predicted Objects')
    
    # Error bars for metrics
    x_pos = np.arange(len(df))
    axes[1, 2].errorbar(x_pos, df['chamfer_distance'], yerr=df['chamfer_distance_std'], 
                       fmt='o', label='Chamfer Distance', alpha=0.7)
    axes[1, 2].set_title('Chamfer Distance with Standard Deviation')
    axes[1, 2].set_xlabel('Case Index')
    axes[1, 2].set_ylabel('Chamfer Distance')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_evaluation_summary(metrics_summary: dict, total_cases: int, successful_cases: int):
    """Print formatted evaluation summary"""
    print(f"\n=== Evaluation Completed ===")
    print(f"Total cases: {total_cases}")
    print(f"Successful cases: {successful_cases}")
    print(f"Failed cases: {total_cases - successful_cases}")
    
    if successful_cases > 0:
        print(f"\n=== Metrics Summary ===")
        print(f"Chamfer Distance: {metrics_summary['chamfer_distance']['mean']:.6f} ± {metrics_summary['chamfer_distance']['std']:.6f}")
        print(f"F-Score: {metrics_summary['f_score']['mean']:.6f} ± {metrics_summary['f_score']['std']:.6f}")
        print(f"IoU: {metrics_summary['iou']['mean']:.6f} ± {metrics_summary['iou']['std']:.6f}")
        print(f"Scene IoU: {metrics_summary['scene_iou']['mean']:.6f} ± {metrics_summary['scene_iou']['std']:.6f}")
        
        # Per-object metrics summary
        if metrics_summary['per_object_metrics']['all_cds']:
            print(f"\n=== Per-Object Metrics Summary ===")
            print(f"Total objects evaluated: {len(metrics_summary['per_object_metrics']['all_cds'])}")
            print(f"Per-object Chamfer Distance: {np.mean(metrics_summary['per_object_metrics']['all_cds']):.6f} ± {np.std(metrics_summary['per_object_metrics']['all_cds']):.6f}")
            print(f"Per-object F-Score: {np.mean(metrics_summary['per_object_metrics']['all_fscores']):.6f} ± {np.std(metrics_summary['per_object_metrics']['all_fscores']):.6f}")
            print(f"Per-object IoU: {np.mean(metrics_summary['per_object_metrics']['all_ious']):.6f} ± {np.std(metrics_summary['per_object_metrics']['all_ious']):.6f}")


def print_case_results(case_name: str, metrics: dict):
    """Print formatted case results"""
    print(f"Case {case_name} evaluation completed")
    print(f"  Chamfer Distance: {metrics['chamfer_distance']:.6f} ± {metrics['chamfer_distance_std']:.6f}")
    print(f"  F-Score: {metrics['f_score']:.6f} ± {metrics['f_score_std']:.6f}")
    print(f"  IoU: {metrics['iou']:.6f} ± {metrics['iou_std']:.6f}")
    print(f"  Scene IoU: {metrics['scene_iou']:.6f}")
    if metrics['per_object_cds']:
        print(f"  Per-object CDs: {[f'{x:.6f}' for x in metrics['per_object_cds']]}")
        print(f"  Per-object F-scores: {[f'{x:.6f}' for x in metrics['per_object_fscores']]}")
        print(f"  Per-object IoUs: {[f'{x:.6f}' for x in metrics['per_object_ious']]}")
