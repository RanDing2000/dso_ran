import argparse
import json
from pathlib import Path
import os
import glob
from datetime import datetime
import subprocess

import numpy as np
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ran.ding/projects/TARGO')

# Grasp planner modules - 专门使用PTV3SceneImplicit
from src.vgn.detection_ptv3_implicit import PTV3SceneGTImplicit

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
    Create a result directory for ptv3_scene model results.
    Then write the command-line arguments to a text file in that directory.
    """
    model_name = 'ptv3_scene_gt'
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


def main(args):
    """
    Main entry point: creates the PTV3SceneImplicit grasp planner, then runs target_sample_offline.
    """
    print("=" * 60)
    print("PTV3 SCENE INFERENCE CONFIGURATION")
    print("=" * 60)
    print(f"Model type: ptv3_scene (fixed)")
    print(f"Model path: {args.model}")
    # print(f"Occlusion level: {args.occlusion_level}")
    print(f"Test root: {args.test_root}")
    print(f"Shape completion: {args.shape_completion}")
    if args.shape_completion:
        print(f"SC model path: {args.sc_model_path}")
    print(f"Hunyuan3D enabled: {args.hunyuan3D_ptv3}")
    if args.hunyuan3D_ptv3:
        print(f"Hunyuan3D path: {args.hunyuan3D_path}")
    else:
        print("Using complete geometry objects")
    print("=" * 60)
    
    # Create PTV3SceneImplicit grasp planner (专门为ptv3_scene设计)
    grasp_planner = PTV3SceneGTImplicit(
        args.model,
        'ptv3_scene_gt',  # 固定使用ptv3_scene
        best=args.best,
        qual_th=args.qual_th,
        force_detection=args.force,
        out_th=args.out_th,
        select_top=False,
        visualize=args.vis,
        sc_model_path=args.sc_model_path if args.shape_completion else None,
        cd_iou_measure=True,
    )
    
    result_path = create_and_write_args_to_result_path(args)
    args.result_path = result_path
    
    # print(f"\nStarting PTV3 Scene inference on {args.occlusion_level} occlusion vgn dataset...")
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
        type='ptv3_scene_gt',  # 确保类型一致
        test_root=args.test_root,
        occ_level_dict_path=args.occ_level_dict,
        hunyun2_path=args.hunyun2_path,
        hunyuan3D_ptv3=args.hunyuan3D_ptv3,
        hunyuan3D_path=args.hunyuan3D_path,
        model_type='ptv3_scene_gt',  # 确保模型类型一致
        video_recording=args.video_recording,
        target_file_path=args.target_file,
        max_scenes=args.max_scenes if hasattr(args, 'max_scenes') else 0,  # 支持限制场景数量
    )

    # Save the result to a JSON file
    result_json_path = f'{args.result_path}/occ_level_sr.json'
    with open(result_json_path, 'w') as f:
        json.dump(occ_level_sr, f, indent=2)
    
    print(f"\nResults saved to: {result_json_path}")
    print(f"Success rate: {occ_level_sr:.4f}")
    
    # Run category analysis after the test is completed
    if os.path.exists(f"{args.result_path}/meta_evaluations.txt"):
        print("\n====== Running category analysis ======")
        analysis_cmd = [
            "python", 
            "targo_eval_results/stastics_analysis/analyze_category_detailed.py", 
            "--eval_file", f"{args.result_path}/meta_evaluations.txt",
            "--output_dir", args.result_path,
            "--data_type", "vgn",
        ]
        print(f"Executing: {' '.join(analysis_cmd)}")
        subprocess.run(analysis_cmd)
        print("====== Category analysis completed ======\n")
    
    print("=" * 60)
    print("PTV3 SCENE INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTV3 Scene Inference on vgn Dataset")

    # Model configuration (simplified for ptv3_scene only)
    parser.add_argument("--model", type=Path, default='/home/ran.ding/projects/TARGO/checkpoints/ptv3_scene.pt',
                        help="Path to the ptv3_scene model checkpoint")
    parser.add_argument("--shape_completion", type=str2bool, default=False,
                        help="Whether to use shape completion during inference")
    parser.add_argument("--sc_model_path", type=str, default=None,
                        help="Path to shape completion model (if shape_completion=True)")
    
    # Dataset configuration
    # parser.add_argument("--occlusion-level", type=str, choices=["no", "slight", "medium"], default="no",
    #                     help="Occlusion level for the experiment: no, slight or medium.")
    parser.add_argument("--test_root", type=str, default='/home/ran.ding/projects/TARGO/data/nips_data_version6/test_set_gaussian_0.002',
                        help="Root directory of test dataset")
    parser.add_argument("--occ_level_dict", type=str, default='/home/ran.ding/projects/TARGO/data/nips_data_version6/test_set_gaussian_0.002/occ_level_dict.json',
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

    # args.result_root = 'targo_eval_results/ycb/eval_results_ptv3_clip-no-occlusion'
    parser.add_argument("--logdir", type=Path, default=None,
                        help="Directory for storing logs or intermediate results.")
    parser.add_argument("--description", type=str, default="ptv3_scene_inference",
                        help="Experiment description.")
    
    # Visualization and debugging
    parser.add_argument("--sim-gui", type=str2bool, default=False,
                        help="Whether to enable a simulation GUI.")
    parser.add_argument("--vis", type=str2bool, default=False,
                        help="Whether to visualize and save the affordance map.")
    parser.add_argument("--video-recording", type=str2bool, default=False,
                        help="Whether to record videos of grasping attempts.")
    parser.add_argument("--target-file", type=str, default='/home/ran.ding/projects/TARGO/example_targets/target_list.txt',
                        help="Path to a .txt file containing target names to record. If provided, only videos of these targets will be recorded.")
    
    # Legacy parameters (kept for compatibility but not used)
    parser.add_argument("--hunyun2_path", type=str, default=None,
                        help="Path to hunyun2 model (not used for ptv3_scene but kept for compatibility).")
    
    # Hunyuan3D support
    parser.add_argument("--hunyuan3D_ptv3", type=str2bool, default=True,
                        help="If True, use Hunyuan3D reconstructed objects; if False, use complete geometry objects")
    parser.add_argument("--hunyuan3D_path", type=str, default=None,
                        help="Path to Hunyuan3D reconstructed objects directory")
    
    args = parser.parse_args()
    args.result_root = 'targo_eval_results/vgn/eval_results/ptv3_scene_gt'

    # Validate model file exists
    if not args.model.exists():
        print(f"ERROR: Model file not found: {args.model}")
        print("Please provide a valid path to ptv3_scene model checkpoint")
        exit(1)
    
    # Validate shape completion configuration
    if args.shape_completion and args.sc_model_path is None:
        print("ERROR: shape_completion=True but sc_model_path not provided")
        print("Please provide --sc_model_path when using shape completion")
        exit(1)
    
    if args.shape_completion and args.sc_model_path and not os.path.exists(args.sc_model_path):
        print(f"ERROR: Shape completion model not found: {args.sc_model_path}")
        exit(1)

   #  Set default logdir if not provided
    if args.logdir is None:
            args.logdir = Path('targo_eval_results/vgn/eval_results/ptv3_scene_gt')

    print("Starting PTV3 Scene inference...")
    main(args)