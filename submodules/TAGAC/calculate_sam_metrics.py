#!/usr/bin/env python3
"""
计算SAM分割结果文件中各项指标的平均值
"""

import os
import pandas as pd
import numpy as np

def calculate_sam_metrics(file_path):
    """计算单个SAM分割结果文件的指标"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    try:
        # 读取CSV文件，处理列名中的空格
        df = pd.read_csv(file_path)
        
        # 清理列名，去除空格
        df.columns = df.columns.str.strip()
        
        # 计算各项指标的平均值
        metrics = {
            'file': os.path.basename(file_path),
            'total_scenes': len(df),
            'avg_iou': df['IoU'].mean(),
            'avg_dice': df['Dice'].mean(),
            'success_rate': df['Success'].mean(),
            'successful_segmentations': df['Success'].sum(),
            'failed_segmentations': len(df) - df['Success'].sum(),
            'min_iou': df['IoU'].min(),
            'max_iou': df['IoU'].max(),
            'min_dice': df['Dice'].min(),
            'max_dice': df['Dice'].max(),
            'std_iou': df['IoU'].std(),
            'std_dice': df['Dice'].std()
        }
        
        return metrics
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def main():
    """主函数"""
    # 三个SAM分割结果文件路径
    files = [
        "targo_eval_results/vgn/eval_results/targo/targo_sam/2025-09-12_16-41-34/sam_segmentation_results.txt",
        "targo_eval_results/vgn/eval_results/targo/targo_sam/2025-09-12_16-42-12/sam_segmentation_results.txt", 
        "targo_eval_results/vgn/eval_results/targo/targo_sam/2025-09-12_16-43-50/sam_segmentation_results.txt"
    ]
    
    print("=" * 80)
    print("SAM分割结果分析")
    print("=" * 80)
    
    all_metrics = []
    
    # 计算每个文件的指标
    for file_path in files:
        print(f"\n处理文件: {os.path.basename(file_path)}")
        print("-" * 60)
        
        metrics = calculate_sam_metrics(file_path)
        if metrics:
            all_metrics.append(metrics)
            
            print(f"总场景数: {metrics['total_scenes']}")
            print(f"成功分割数: {metrics['successful_segmentations']}")
            print(f"失败分割数: {metrics['failed_segmentations']}")
            print(f"分割成功率: {metrics['success_rate']:.4f} ({metrics['success_rate']*100:.2f}%)")
            print(f"平均IoU: {metrics['avg_iou']:.4f}")
            print(f"平均Dice: {metrics['avg_dice']:.4f}")
            print(f"IoU范围: [{metrics['min_iou']:.4f}, {metrics['max_iou']:.4f}]")
            print(f"Dice范围: [{metrics['min_dice']:.4f}, {metrics['max_dice']:.4f}]")
            print(f"IoU标准差: {metrics['std_iou']:.4f}")
            print(f"Dice标准差: {metrics['std_dice']:.4f}")
    
    # 计算总体统计
    if all_metrics:
        print("\n" + "=" * 80)
        print("总体统计")
        print("=" * 80)
        
        total_scenes = sum(m['total_scenes'] for m in all_metrics)
        total_successful = sum(m['successful_segmentations'] for m in all_metrics)
        total_failed = sum(m['failed_segmentations'] for m in all_metrics)
        
        # 加权平均IoU和Dice
        weighted_avg_iou = sum(m['avg_iou'] * m['total_scenes'] for m in all_metrics) / total_scenes
        weighted_avg_dice = sum(m['avg_dice'] * m['total_scenes'] for m in all_metrics) / total_scenes
        
        overall_success_rate = total_successful / total_scenes
        
        print(f"总场景数: {total_scenes}")
        print(f"总成功分割数: {total_successful}")
        print(f"总失败分割数: {total_failed}")
        print(f"总体分割成功率: {overall_success_rate:.4f} ({overall_success_rate*100:.2f}%)")
        print(f"加权平均IoU: {weighted_avg_iou:.4f}")
        print(f"加权平均Dice: {weighted_avg_dice:.4f}")
        
        # 各文件间的比较
        print("\n" + "=" * 80)
        print("各文件间比较")
        print("=" * 80)
        
        print(f"{'文件名':<50} {'成功率':<10} {'平均IoU':<10} {'平均Dice':<10}")
        print("-" * 80)
        
        for metrics in all_metrics:
            filename = metrics['file'][:47] + "..." if len(metrics['file']) > 50 else metrics['file']
            print(f"{filename:<50} {metrics['success_rate']*100:>8.2f}% {metrics['avg_iou']:>9.4f} {metrics['avg_dice']:>9.4f}")
        
        # 找出最佳和最差表现
        best_success = max(all_metrics, key=lambda x: x['success_rate'])
        worst_success = min(all_metrics, key=lambda x: x['success_rate'])
        best_iou = max(all_metrics, key=lambda x: x['avg_iou'])
        best_dice = max(all_metrics, key=lambda x: x['avg_dice'])
        
        print(f"\n最佳分割成功率: {best_success['file']} ({best_success['success_rate']*100:.2f}%)")
        print(f"最差分割成功率: {worst_success['file']} ({worst_success['success_rate']*100:.2f}%)")
        print(f"最佳平均IoU: {best_iou['file']} ({best_iou['avg_iou']:.4f})")
        print(f"最佳平均Dice: {best_dice['file']} ({best_dice['avg_dice']:.4f})")
        
        # 保存结果到CSV
        results_df = pd.DataFrame(all_metrics)
        output_file = "sam_metrics_summary.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
