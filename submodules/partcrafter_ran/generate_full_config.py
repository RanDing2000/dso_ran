#!/usr/bin/env python3
import os
import json
import glob

def generate_full_config():
    """生成包含所有preprocessed_data子文件夹的完整配置文件"""
    
    # 基础路径
    base_path = "preprocessed_data"
    output_file = "datasets/object_part_configs_data_full.json"
    
    # 获取所有子文件夹
    subdirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item not in ["sword", "scissors"]:
            subdirs.append(item)
    
    print(f"找到 {len(subdirs)} 个子文件夹")
    
    # 生成配置数据
    config_data = []
    skipped_count = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_path, subdir)
        
        # 检查必需文件是否存在
        points_file = os.path.join(subdir_path, "points.npy")
        rendering_file = os.path.join(subdir_path, "rendering_rmbg.png")
        num_parts_file = os.path.join(subdir_path, "num_parts.json")
        iou_file = os.path.join(subdir_path, "iou.json")
        
        if not all(os.path.exists(f) for f in [points_file, rendering_file, num_parts_file, iou_file]):
            skipped_count += 1
            continue
        
        # 读取num_parts.json
        try:
            with open(num_parts_file, 'r') as f:
                num_parts_data = json.load(f)
                num_parts = num_parts_data.get("num_parts", 1)
        except:
            num_parts = 1
        
        # 读取iou.json
        try:
            with open(iou_file, 'r') as f:
                iou_data = json.load(f)
                iou_mean = iou_data.get("iou_mean", 0.0)
                iou_max = iou_data.get("iou_max", 0.0)
        except:
            iou_mean = 0.0
            iou_max = 0.0
        
        # 创建配置项
        config_item = {
            "file": f"{subdir}.glb",  # 假设文件名
            "num_parts": num_parts,
            "valid": True,
            "mesh_path": f"assets/objects/{subdir}.glb",  # 假设路径
            "surface_path": f"/data3/ran/preprocessed_data/{subdir}/points.npy",
            "image_path": f"/data3/ran/preprocessed_data/{subdir}/rendering_rmbg.png",
            "iou_mean": iou_mean,
            "iou_max": iou_max
        }
        
        config_data.append(config_item)
    
    print(f"跳过了 {skipped_count} 个不完整的文件夹")
    print(f"生成了 {len(config_data)} 个配置项")
    
    # 写入文件
    with open(output_file, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    print(f"配置文件已保存到: {output_file}")

if __name__ == "__main__":
    generate_full_config()
