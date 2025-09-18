#!/usr/bin/env python3
"""
TARGO Inference with SAM (Segment Anything Model) for ablation study.
This script uses SAM to segment target objects instead of using ground truth masks.
"""

import argparse
import json
from pathlib import Path
import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import torch
import cv2
import random

# Add project root to path
sys.path.append('/home/ran.ding/projects/TARGO')

# SAM imports
from segment_anything import sam_model_registry, SamPredictor

# Grasp planner modules
from src.vgn.detection_implicit_vgn import VGNImplicit
from src.vgn.experiments import target_sample_offline_vgn

# Utility
from src.vgn.utils.misc import set_random_seed

def str2bool(v):
    """Convert string inputs like 'yes', 'true', etc. to boolean values."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_and_write_args_to_result_path(args):
    """Create a result directory for model results based on model type."""
    # Create SAM-specific result directory
    model_name = f"{args.model_type}_sam"
    result_directory = f'{args.result_root}/{model_name}'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_filename = f"{timestamp}"
    result_file_path = os.path.join(result_directory, result_filename)
    args.result_path = result_file_path

    args_dict = vars(args)
    args_content = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())

    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)

    if args.vis:
        args.logdir = Path(result_file_path)

    result_initial_path = f'{result_file_path}/initial.txt'
    with open(result_initial_path, 'w') as result_file:
        result_file.write(args_content)

    print(f"Args saved to {result_initial_path}")
    return result_file_path


def calculate_segmentation_success_rate(result_path):
    """Calculate SAM segmentation success rate from sam_segmentation_results.txt"""
    
    sam_results_path = os.path.join(result_path, 'sam_segmentation_results.txt')
    
    if not os.path.exists(sam_results_path):
        print(f"Warning: sam_segmentation_results.txt not found at {sam_results_path}")
        return None
    
    try:
        total_scenes = 0
        successful_segmentations = 0
        segmentation_metrics = []
        
        with open(sam_results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('Scene_ID'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        try:
                            scene_id = parts[0]
                            iou = float(parts[1])
                            dice = float(parts[2])
                            success = int(parts[3])
                            
                            total_scenes += 1
                            if success == 1:
                                successful_segmentations += 1
                            
                            segmentation_metrics.append({
                                'scene_id': scene_id,
                                'iou': iou,
                                'dice': dice,
                                'success': success
                            })
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line: {line} - {e}")
                            continue
        
        if total_scenes > 0:
            success_rate = successful_segmentations / total_scenes
            avg_iou = sum(m['iou'] for m in segmentation_metrics) / len(segmentation_metrics)
            avg_dice = sum(m['dice'] for m in segmentation_metrics) / len(segmentation_metrics)
            
            print(f"=== SAM SEGMENTATION RESULTS ===")
            print(f"Total scenes: {total_scenes}")
            print(f"Successful segmentations: {successful_segmentations}")
            print(f"Segmentation success rate: {success_rate:.4f} ({success_rate*100:.2f}%)")
            print(f"Average IoU: {avg_iou:.4f}")
            print(f"Average Dice: {avg_dice:.4f}")
            print("=" * 35)
            
            return {
                'total_scenes': total_scenes,
                'successful_segmentations': successful_segmentations,
                'success_rate': success_rate,
                'avg_iou': avg_iou,
                'avg_dice': avg_dice,
                'segmentation_metrics': segmentation_metrics
            }
        else:
            print("No segmentation data found")
            return None
            
    except Exception as e:
        print(f"Error calculating segmentation success rate: {e}")
        return None


def generate_filtered_result_csv(result_path, model_type):
    """Generate filtered_result.csv based on meta_evaluations.txt file."""
    
    meta_eval_path = os.path.join(result_path, 'meta_evaluations.txt')
    
    if not os.path.exists(meta_eval_path):
        print(f"Warning: meta_evaluations.txt not found at {meta_eval_path}")
        return
    
    try:
        # Read the meta_evaluations.txt file
        results = []
        with open(meta_eval_path, 'r') as f:
            lines = f.readlines()
            
        # Parse the evaluation data
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Scene_ID') and not line.startswith('Average') and not line.startswith('Success') and not line.startswith('Total') and not line.startswith('Successful') and line != '':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    try:
                        scene_id = parts[0]
                        occlusion_level = float(parts[1])
                        num_occluders = int(parts[2])
                        length = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        success = int(parts[6])
                        # Comment out IoU and CD calculation for SAM ablation
                        # iou = float(parts[7])
                        # cd = float(parts[8])
                        iou = 0.0  # Default value for SAM ablation
                        cd = 0.0   # Default value for SAM ablation
                        
                        length_width = min(length, width)
                        
                        results.append({
                            'scene_id': scene_id,
                            'success': success,
                            'occlusion_level': occlusion_level,
                            'length_width': length_width,
                            'length': length,
                            'width': width,
                            'height': height,
                            'num_occluders': num_occluders,
                            'iou': iou,
                            'cd': cd
                        })
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line: {line} - {e}")
                        continue
        
        if not results:
            print("No valid data found in meta_evaluations.txt")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Create bins for length_width and occlusion_level
        length_width_bins = [(0.0166, 0.0277), (0.0277, 0.0393), (0.0393, 0.0509), 
                           (0.0509, 0.0626), (0.0626, 0.0742)]
        occlusion_bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                         (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]
        
        # Create binned data
        filtered_results = []
        
        for lw_bin in length_width_bins:
            for occ_bin in occlusion_bins:
                for num_occ in [3, 4, 5]:
                    # Filter data for this bin combination
                    mask = (
                        (df['length_width'] > lw_bin[0]) & (df['length_width'] <= lw_bin[1]) &
                        (df['occlusion_level'] >= occ_bin[0]) & (df['occlusion_level'] < occ_bin[1]) &
                        (df['num_occluders'] == num_occ)
                    )
                    
                    bin_data = df[mask]
                    
                    if len(bin_data) > 0:
                        success_rate = bin_data['success'].mean()
                        count = len(bin_data)
                        
                        lw_bin_str = f"({lw_bin[0]}, {lw_bin[1]}]"
                        occ_bin_str = f"[{occ_bin[0]}, {occ_bin[1]})"
                        
                        filtered_results.append({
                            'length_width_bin': lw_bin_str,
                            'occlusion_bin': occ_bin_str,
                            'num_occluders': num_occ,
                            'success_rate': success_rate,
                            'count': count
                        })
        
        # Create DataFrame and save to CSV
        filtered_df = pd.DataFrame(filtered_results)
        csv_path = os.path.join(result_path, 'filtered_result.csv')
        filtered_df.to_csv(csv_path, index=False, quoting=1)
        
        print(f"Generated filtered_result.csv at: {csv_path}")
        print(f"Total bins with data: {len(filtered_results)}")
        
        # Also save a detailed summary
        summary_path = os.path.join(result_path, 'result_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"=== EVALUATION SUMMARY (SAM VERSION) ===\n")
            f.write(f"Model Type: {model_type}_sam\n")
            f.write(f"Total Evaluations: {len(results)}\n")
            f.write(f"Overall Success Rate: {df['success'].mean():.4f}\n")
            f.write(f"Average Length/Width: {df['length_width'].mean():.6f}\n")
            f.write(f"Average Length: {df['length'].mean():.6f}\n")
            f.write(f"Average Width: {df['width'].mean():.6f}\n")
            f.write(f"Average Height: {df['height'].mean():.6f}\n")
            # Comment out IoU and CD for SAM ablation
            # f.write(f"Average IoU: {df['iou'].mean():.6f}\n")
            # f.write(f"Average CD: {df['cd'].mean():.6f}\n")
            f.write(f"Average IoU: 0.000000 (SAM ablation - not calculated)\n")
            f.write(f"Average CD: 0.000000 (SAM ablation - not calculated)\n")
            f.write(f"Total Bins with Data: {len(filtered_results)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Generated summary at: {summary_path}")
        
    except Exception as e:
        print(f"Error generating filtered_result.csv: {e}")


def main(args):
    """Main entry point: creates the VGNImplicit grasp planner with SAM, then runs target_sample_offline."""
    print("=" * 60)
    print("TARGO INFERENCE WITH SAM CONFIGURATION")
    print("=" * 60)
    print(f"Model type: {args.model_type}_sam")
    print(f"Model path: {args.model}")
    print(f"SAM model: {args.sam_model_type}")
    print(f"SAM checkpoint: {args.sam_checkpoint}")
    print(f"Occlusion level: {args.occlusion_level}")
    print(f"Test root: {args.test_root}")
    
    # Shape completion logic based on model type
    use_shape_completion = args.shape_completion and args.model_type == "targo"
    print(f"Shape completion: {use_shape_completion}")
    if use_shape_completion:
        print(f"SC model path: {args.sc_model_path}")
    elif args.model_type in ["targo_partial", "targo_full_gt"]:
        print("Shape completion disabled for this model type")
        
    print(f"Hunyuan3D enabled: {args.hunyuan3D}")
    if args.hunyuan3D:
        print(f"Hunyuan3D path: {args.hunyuan3D_path}")
    else:
        print("Using complete geometry objects")
    print("=" * 60)
    
    # Create VGNImplicit grasp planner
    grasp_planner = VGNImplicit(
        args.model,
        args.model_type,
        best=args.best,
        qual_th=args.qual_th,
        force_detection=args.force,
        out_th=args.out_th,
        select_top=False,
        visualize=args.vis,
        sc_model_path=args.sc_model_path if use_shape_completion else None,
        cd_iou_measure=True,
    )
    
    result_path = create_and_write_args_to_result_path(args)
    args.result_path = result_path
    
    print(f"\nStarting TARGO inference with SAM on {args.occlusion_level} occlusion vgn dataset...")
    print(f"Result path: {result_path}")
    
    # Run target_sample_offline evaluation with SAM
    occ_level_sr = target_sample_offline_vgn.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        sim_gui=args.sim_gui,
        result_path=args.result_path,
        add_noise=None,
        sideview=args.sideview,
        visualize=args.vis,
        type=args.model_type,
        test_root=args.test_root,
        occ_level_dict_path=args.occ_level_dict,
        model_type=args.model_type,
        video_recording=args.video_recording,
        target_file_path=args.target_file,
        max_scenes=args.max_scenes,
        sc_net=grasp_planner.sc_net,
        vis_failure_only=args.vis_failure_only,
        vis_rgb=args.vis_rgb,
        vis_gt_target=args.vis_gt_target,
        vis_grasps=args.vis_grasps,
        vis_target_mesh_pred=args.vis_target_mesh_pred,
        # SAM-specific parameters
        use_sam=True,
        sam_model_type=args.sam_model_type,
        sam_checkpoint=args.sam_checkpoint,
    )

    # Save the result to a JSON file
    result_json_path = f'{args.result_path}/occ_level_sr.json'
    with open(result_json_path, 'w') as f:
        json.dump(occ_level_sr, f, indent=2)
    
    print(f"\nResults saved to: {result_json_path}")
    print(f"Success rate: {occ_level_sr:.4f}")
    
    # Generate filtered_result.csv after the test is completed
    if os.path.exists(f"{args.result_path}/meta_evaluations.txt"):
        print("\n====== Generating filtered result CSV ======")
        generate_filtered_result_csv(args.result_path, args.model_type)
        print("====== Filtered result CSV generated ======\n")
    
    # Calculate SAM segmentation success rate
    print("\n====== Calculating SAM Segmentation Success Rate ======")
    segmentation_results = calculate_segmentation_success_rate(args.result_path)
    if segmentation_results:
        # Save segmentation results to JSON
        seg_results_path = f'{args.result_path}/sam_segmentation_summary.json'
        with open(seg_results_path, 'w') as f:
            json.dump(segmentation_results, f, indent=2)
        print(f"SAM segmentation results saved to: {seg_results_path}")
    else:
        print("No SAM segmentation results found")
    print("====== SAM Segmentation Analysis Complete ======\n")
    
    print("=" * 60)
    print("TARGO INFERENCE WITH SAM COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARGO Inference with SAM on vgn Dataset")

    # Model configuration
    parser.add_argument("--model", type=Path, default='checkpoints/targonet.pt',
                        help="Path to the targo model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["targo", "targo_partial", "targo_full_gt"], default="targo",
                        help="Model type: targo (with shape completion), targo_partial (partial input), targo_full_gt (ground truth complete target)")
    parser.add_argument("--shape_completion", type=str2bool, default=True,
                        help="Whether to use shape completion during inference")
    parser.add_argument("--sc_model_path", type=str, default='checkpoints/adapointr.pth',
                        help="Path to shape completion model (if shape_completion=True)")
    
    # SAM configuration
    parser.add_argument("--sam_model_type", type=str, choices=["vit_h", "vit_l", "vit_b"], default="vit_b",
                        help="SAM model type: vit_h (default), vit_l, or vit_b")
    parser.add_argument("--sam_checkpoint", type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help="Path to SAM model checkpoint")
    
    # Dataset configuration
    parser.add_argument("--occlusion-level", type=str, choices=["no", "slight", "medium"], default="no",
                        help="Occlusion level for the experiment: no, slight or medium.")
    parser.add_argument("--test_root", type=str, default='data/nips_data_version6/test_set_gaussian_0.002',
                        help="Root directory of test dataset")
    parser.add_argument("--occ_level_dict", type=str, default='data/nips_data_version6/test_set_gaussian_0.002/occ_level_dict.json',
                        help="Path to occlusion level dictionary JSON file")
    
    # Inference parameters
    parser.add_argument("--out_th", type=float, default=0.15,
                        help="Output threshold for valid grasps.")
    parser.add_argument("--qual-th", type=float, default=0.9,
                        help="Quality threshold for valid grasps.")
    parser.add_argument("--best", type=str2bool, default=True,
                        help="Use the best valid grasp if available.")
    parser.add_argument("--force", type=str2bool, default=True,
                        help="Force selection of a grasp even if below threshold.")
    parser.add_argument("--max_scenes", type=int, default=150,
                        help="Maximum number of scenes to process (0 for all)")
    
    # Scene configuration
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--sideview", type=str2bool, default=True,
                        help="Capture the scene from one side rather than top-down.")
    
    # Output configuration
    parser.add_argument("--result_root", type=Path, default=None,
                        help="Root directory for saving results")
    parser.add_argument("--logdir", type=Path, default=None,
                        help="Directory for storing logs or intermediate results.")
    parser.add_argument("--description", type=str, default="targo_sam_inference",
                        help="Experiment description.")
    
    # Visualization and debugging
    parser.add_argument("--sim-gui", type=str2bool, default=False,
                        help="Whether to enable a simulation GUI.")
    parser.add_argument("--vis", type=str2bool, default=False,
                        help="Whether to visualize and save the affordance map.")
    parser.add_argument("--vis_failure_only", type=str2bool, default=False,
                        help="Whether to visualize only failure cases.")
    parser.add_argument("--vis_rgb", type=str2bool, default=True,
                        help="Whether to visualize RGB images.")
    parser.add_argument("--vis_gt_target", type=str2bool, default=True,
                        help="Whether to visualize GT target mesh.")
    parser.add_argument("--vis_grasps", type=str2bool, default=True,
                        help="Whether to visualize grasps on affordance mesh.")
    parser.add_argument("--vis_target_mesh_pred", type=str2bool, default=True,
                        help="Whether to save predicted target mesh.")
    parser.add_argument("--video-recording", type=str2bool, default=False,
                        help="Whether to record videos of grasping attempts.")
    parser.add_argument("--target-file", type=str, default='/home/ran.ding/projects/TARGO/example_targets/target_list.txt',
                        help="Path to a .txt file containing target names to record.")
    
    # Legacy parameters (kept for compatibility but not used)
    parser.add_argument("--hunyun2_path", type=str, default=None,
                        help="Path to hunyun2 model (not used for targo but kept for compatibility).")
    
    # Hunyuan3D support
    parser.add_argument("--hunyuan3D", type=str2bool, default=True,
                        help="If True, use Hunyuan3D reconstructed objects; if False, use complete geometry objects")
    parser.add_argument("--hunyuan3D_path", type=str, default=None,
                        help="Path to Hunyuan3D reconstructed objects directory")
    
    args = parser.parse_args()
    args.result_root = 'targo_eval_results/vgn/eval_results/targo'

    # Validate model file exists
    if not args.model.exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("Please provide a valid path to targo model checkpoint")
        exit(1)
    
    # Validate SAM checkpoint exists
    if not os.path.exists(args.sam_checkpoint):
        print(f"ERROR: SAM checkpoint not found: {args.sam_checkpoint}")
        print("Please provide a valid path to SAM model checkpoint")
        print("Download SAM checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
        exit(1)
    
    # Validate shape completion configuration
    if args.model_type == "targo" and args.shape_completion and args.sc_model_path is None:
        print("ERROR: shape_completion=True but sc_model_path not provided for targo model")
        print("Please provide --sc_model_path when using shape completion")
        exit(1)
    
    if args.model_type == "targo" and args.shape_completion and args.sc_model_path and not os.path.exists(args.sc_model_path):
        print(f"ERROR: Shape completion model not found: {args.sc_model_path}")
        exit(1)
    
    # For targo_partial and targo_full_gt, shape completion is not used
    if args.model_type in ["targo_partial", "targo_full_gt"]:
        args.shape_completion = False
        print(f"Shape completion automatically disabled for {args.model_type} model type")

    # Set default logdir if not provided
    if args.logdir is None:
        args.logdir = Path('targo_eval_results/vgn/eval_results/targo')

    print("Starting TARGO inference with SAM...")
    main(args)
