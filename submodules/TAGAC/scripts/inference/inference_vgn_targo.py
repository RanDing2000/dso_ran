import argparse
import json
from pathlib import Path
import os
import glob
from datetime import datetime
import pandas as pd

import numpy as np
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ran.ding/projects/TARGO')

# Grasp planner modules - 专门使用VGNImplicit
from src.vgn.detection_implicit_vgn import VGNImplicit

# Only keep target_sample_offline
from src.vgn.experiments import target_sample_offline_vgn

# Utility
from src.vgn.utils.misc import set_random_seed

def str2bool(v):
    """
    Convert string inputs like 'yes', 'true', etc. to boolean values.
    Raises an error for invalid inputs.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_and_write_args_to_result_path(args):
    """
    Create a result directory for model results based on model type.
    Then write the command-line arguments to a text file in that directory.
    """
    # 根据模型类型创建对应的子文件夹
    model_name = args.model_type  # 使用 targo_partial, targo_full_gt, 或 targo
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


def generate_filtered_result_csv(result_path, model_type):
    """
    Generate filtered_result.csv based on meta_evaluations.txt file.
    This function analyzes the evaluation results and creates bins for different metrics.
    """
    
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
                # Parse each line to extract relevant information
                # New Format: Scene_ID, Occlusion_Level, Num_Occluders, Length, Width, Height, Success, IoU, CD
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:  # Updated to handle IoU and CD columns
                    try:
                        scene_id = parts[0]
                        occlusion_level = float(parts[1])
                        num_occluders = int(parts[2])
                        length = float(parts[3])
                        width = float(parts[4])
                        height = float(parts[5])
                        success = int(parts[6])
                        iou = float(parts[7])
                        cd = float(parts[8])
                        
                        # Calculate length_width as the minimum of length and width (horizontal dimensions)
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
        
        # Debug: Show comprehensive data distribution
        print(f"=== DATA DISTRIBUTION ANALYSIS ===")
        print(f"Total samples parsed: {len(results)}")
        
        # Length_width distribution
        print(f"\nLength_width distribution:")
        length_width_counts = {}
        for lw in df['length_width']:
            for i, (low, high) in enumerate([(0.0166, 0.0277), (0.0277, 0.0393), (0.0393, 0.0509), 
                                           (0.0509, 0.0626), (0.0626, 0.0742)]):
                if low < lw <= high:
                    bin_name = f"({low}, {high}]"
                    length_width_counts[bin_name] = length_width_counts.get(bin_name, 0) + 1
                    break
        for bin_name, count in sorted(length_width_counts.items()):
            print(f"  {bin_name}: {count} samples")
        
        # Occlusion level distribution
        print(f"\nOcclusion level distribution:")
        occ_counts = {}
        for occ in df['occlusion_level']:
            for low, high in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                             (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]:
                if low <= occ < high:
                    bin_name = f"[{low}, {high})"
                    occ_counts[bin_name] = occ_counts.get(bin_name, 0) + 1
                    break
        for bin_name, count in sorted(occ_counts.items()):
            print(f"  {bin_name}: {count} samples")
        
        # Number of occluders distribution
        print(f"\nNumber of occluders distribution:")
        occluder_counts = df['num_occluders'].value_counts().sort_index()
        for num_occ, count in occluder_counts.items():
            print(f"  {num_occ} occluders: {count} samples")
        
        print(f"=== END ANALYSIS ===\n")
        
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
                        
                        # Debug: Print bin details for comparison
                        lw_bin_str = f"({lw_bin[0]}, {lw_bin[1]}]"
                        occ_bin_str = f"[{occ_bin[0]}, {occ_bin[1]})"
                        print(f"Bin: {lw_bin_str}, {occ_bin_str}, {num_occ} occluders -> {count} samples, SR: {success_rate:.4f}")
                        
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
        # 确保CSV格式与参考文件一致，字符串用双引号括起来
        filtered_df.to_csv(csv_path, index=False, quoting=1)  # quoting=1 means QUOTE_ALL
        
        print(f"Generated filtered_result.csv at: {csv_path}")
        print(f"Total bins with data: {len(filtered_results)}")
        
        # Also save a detailed summary
        summary_path = os.path.join(result_path, 'result_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"=== EVALUATION SUMMARY ===\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Total Evaluations: {len(results)}\n")
            f.write(f"Overall Success Rate: {df['success'].mean():.4f}\n")
            f.write(f"Average Length/Width: {df['length_width'].mean():.6f}\n")
            f.write(f"Average Length: {df['length'].mean():.6f}\n")
            f.write(f"Average Width: {df['width'].mean():.6f}\n")
            f.write(f"Average Height: {df['height'].mean():.6f}\n")
            f.write(f"Average IoU: {df['iou'].mean():.6f}\n")
            f.write(f"Average CD: {df['cd'].mean():.6f}\n")
            f.write(f"Total Bins with Data: {len(filtered_results)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add breakdown by occlusion levels
            f.write("=== BREAKDOWN BY OCCLUSION LEVEL ===\n")
            for occ_bin in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                           (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]:
                mask = (df['occlusion_level'] >= occ_bin[0]) & (df['occlusion_level'] < occ_bin[1])
                occ_data = df[mask]
                if len(occ_data) > 0:
                    f.write(f"Occlusion [{occ_bin[0]:.1f}, {occ_bin[1]:.1f}): {occ_data['success'].mean():.4f} ({len(occ_data)} scenes)\n")
            
            # Add breakdown by number of occluders
            f.write("\n=== BREAKDOWN BY NUMBER OF OCCLUDERS ===\n")
            for num_occ in [3, 4, 5]:
                mask = df['num_occluders'] == num_occ
                occ_data = df[mask]
                if len(occ_data) > 0:
                    f.write(f"{num_occ} occluders: {occ_data['success'].mean():.4f} ({len(occ_data)} scenes)\n")
        
        print(f"Generated summary at: {summary_path}")
        
    except Exception as e:
        print(f"Error generating filtered_result.csv: {e}")


def main(args):
    """
    Main entry point: creates the VGNImplicit grasp planner, then runs target_sample_offline.
    """
    print("=" * 60)
    print("TARGO INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Model path: {args.model}")
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
        args.model_type,  # 使用指定的模型类型
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
    
    print(f"\nStarting TARGO inference on {args.occlusion_level} occlusion vgn dataset...")
    print(f"Result path: {result_path}")
    
    # Run target_sample_offline evaluation (与训练脚本保持一致的调用方式)
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
        # type='targo',  # 确保类型一致
        type = args.model_type,
        test_root=args.test_root,
        occ_level_dict_path=args.occ_level_dict,
        model_type=args.model_type,  # 使用指定的模型类型
        video_recording=args.video_recording,
        target_file_path=args.target_file,
        max_scenes=args.max_scenes,  # 处理场景数量限制（0表示处理所有场景）
        sc_net=grasp_planner.sc_net,  # 传递shape completion network
        vis_failure_only=args.vis_failure_only,  # 可视化失败案例
        vis_rgb=args.vis_rgb,  # 可视化RGB图像
        vis_gt_target=args.vis_gt_target,  # 可视化GT target mesh
        vis_grasps=args.vis_grasps,  # 可视化grasps
        vis_target_mesh_pred=args.vis_target_mesh_pred,  # 保存预测的target mesh
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
    
    print("=" * 60)
    print("TARGO INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARGO Inference on vgn Dataset")

    # Model configuration (simplified for targo only)
    parser.add_argument("--model", type=Path, default='checkpoints/targonet.pt',
                        help="Path to the targo model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["targo", "targo_partial", "targo_full_gt"], default="targo_partial",
                        help="Model type: targo (with shape completion), targo_partial (partial input), targo_full_gt (ground truth complete target)")
    parser.add_argument("--shape_completion", type=str2bool, default=True,
                        help="Whether to use shape completion during inference")
    parser.add_argument("--sc_model_path", type=str, default='checkpoints/adapointr.pth',
                        help="Path to shape completion model (if shape_completion=True)")
    
    # Dataset configuration
    parser.add_argument("--occlusion-level", type=str, choices=["no", "slight", "medium"], default="no",
                        help="Occlusion level for the experiment: no, slight or medium.")
    parser.add_argument("--test_root", type=str, default='data/nips_data_version6/test_set_gaussian_0.002',
                        help="Root directory of test dataset")
    parser.add_argument("--occ_level_dict", type=str, default='data/nips_data_version6/test_set_gaussian_0.002/occ_level_dict.json',
                        help="Path to occlusion level dictionary JSON file")
    
    # Inference parameters
    parser.add_argument("--out_th", type=float, default=0.2,
                        help="Output threshold for valid grasps.")
    parser.add_argument("--qual-th", type=float, default=0.9,
                        help="Quality threshold for valid grasps.")
    parser.add_argument("--best", type=str2bool, default=True,
                        help="Use the best valid grasp if available.")
    parser.add_argument("--force", type=str2bool, default=True,
                        help="Force selection of a grasp even if below threshold.")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Maximum number of scenes to process (0 for all)")
    
    # Scene configuration
    parser.add_argument("--scene", type=str, choices=["pile", "packed"], default="packed")
    parser.add_argument("--object-set", type=str, default="packed/train")
    parser.add_argument("--sideview", type=str2bool, default=True,
                        help="Capture the scene from one side rather than top-down.")
    
    # Output configuration
    parser.add_argument("--result_root", type=Path, default=None,
                        help="Root directory for saving results")

    # args.result_root = 'targo_eval_results/ycb/eval_results_targo-no-occlusion'
    parser.add_argument("--logdir", type=Path, default=None,
                        help="Directory for storing logs or intermediate results.")
    parser.add_argument("--description", type=str, default="targo_inference",
                        help="Experiment description.")
    
    # Visualization and debugging
    parser.add_argument("--sim-gui", type=str2bool, default=False,
                        help="Whether to enable a simulation GUI.")
    parser.add_argument("--vis", type=str2bool, default=True,
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
                        help="Path to a .txt file containing target names to record. If provided, only videos of these targets will be recorded.")
    
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

   #  Set default logdir if not provided
    if args.logdir is None:
            args.logdir = Path('targo_eval_results/vgn/eval_results/targo')

    print("Starting TARGO inference...")
    main(args)
