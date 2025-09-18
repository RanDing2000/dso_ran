#!/usr/bin/env python3
"""
Demo script showing how to use existing results for evaluation
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

def demo_existing_results_usage():
    """Demonstrate how to use existing results for evaluation"""
    
    print("="*60)
    print("PartCrafter 评估脚本 - 使用现有结果演示")
    print("="*60)
    
    print("\n1. 功能说明:")
    print("   - 自动检测现有预测结果文件")
    print("   - 跳过推理步骤，节省时间")
    print("   - 仍然进行对齐和计算指标")
    print("   - 支持强制重新推理")
    
    print("\n2. 支持的文件格式:")
    print("   - gt_merged.glb: 合并的GT网格")
    print("   - pred_merged.glb: 合并的预测网格")
    print("   - gt_part_*.glb: 单个GT部件")
    print("   - pred_part_*.glb: 单个预测部件")
    
    print("\n3. 使用示例:")
    
    print("\n   示例1: 使用现有结果进行评估（推荐）")
    print("   python scripts/eval.py \\")
    print("       --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \\")
    print("       --output_dir results/evaluation_messy_kitchen \\")
    print("       --use_existing_results \\")
    print("       --num_samples 10000")
    
    print("\n   示例2: 强制重新运行推理")
    print("   python scripts/eval.py \\")
    print("       --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \\")
    print("       --output_dir results/evaluation_messy_kitchen \\")
    print("       --force_inference \\")
    print("       --num_samples 10000")
    
    print("\n   示例3: 禁用使用现有结果")
    print("   python scripts/eval.py \\")
    print("       --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \\")
    print("       --output_dir results/evaluation_messy_kitchen \\")
    print("       --no-use_existing_results \\")
    print("       --num_samples 10000")
    
    print("\n4. 预期输出:")
    print("   使用现有结果时:")
    print("   Evaluating case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed")
    print("   Using existing results for case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed")
    print("   Loaded 6 predicted meshes and 6 GT meshes from existing results")
    print("   Computing metrics with alignment for 6 predicted meshes...")
    
    print("\n   运行推理时:")
    print("   Evaluating case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed")
    print("   Running inference for case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed")
    print("   Inference time: 45.23 seconds")
    print("   Computing metrics with alignment for 6 predicted meshes...")
    
    print("\n5. 优势:")
    print("   - 节省时间: 跳过推理步骤")
    print("   - 节省资源: 减少GPU内存使用")
    print("   - 保持一致性: 使用相同的预测结果")
    print("   - 灵活控制: 可选择是否使用现有结果")
    
    print("\n6. 注意事项:")
    print("   - 确保现有结果文件完整且未损坏")
    print("   - 检查文件命名格式是否正确")
    print("   - 验证文件权限")
    print("   - 对齐和指标计算仍会正常进行")

def check_existing_results_structure():
    """Check the structure of existing results"""
    
    print("\n" + "="*60)
    print("检查现有结果文件结构")
    print("="*60)
    
    # 检查示例目录
    example_dir = "results/evaluation_messy_kitchen/bb7c492421494988a9abfd8e1accb0cd_combined_fixed"
    
    if os.path.exists(example_dir):
        print(f"✓ 找到示例目录: {example_dir}")
        
        files = os.listdir(example_dir)
        print(f"  包含 {len(files)} 个文件:")
        
        # 分类文件
        pred_files = [f for f in files if f.startswith('pred_')]
        gt_files = [f for f in files if f.startswith('gt_')]
        other_files = [f for f in files if not f.startswith(('pred_', 'gt_'))]
        
        print(f"  预测文件 ({len(pred_files)}): {pred_files}")
        print(f"  GT文件 ({len(gt_files)}): {gt_files}")
        print(f"  其他文件 ({len(other_files)}): {other_files}")
        
        # 检查关键文件
        key_files = ['gt_merged.glb', 'pred_merged.glb', 'mesh_summary.json']
        for key_file in key_files:
            file_path = os.path.join(example_dir, key_file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  ✓ {key_file} ({size:.1f} MB)")
            else:
                print(f"  ✗ {key_file} (缺失)")
    else:
        print(f"✗ 示例目录不存在: {example_dir}")
        print("  请先运行一次完整的评估来生成示例结果")

if __name__ == "__main__":
    demo_existing_results_usage()
    check_existing_results_structure()
