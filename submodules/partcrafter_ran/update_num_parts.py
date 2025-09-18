#!/usr/bin/env python3
import json
import os
from pathlib import Path

def update_config_file(config_file_path):
    """更新配置文件中的num_parts字段"""
    print(f"正在处理配置文件: {config_file_path}")
    
    # 读取配置文件
    with open(config_file_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    updated_count = 0
    error_count = 0
    
    for i, config in enumerate(configs):
        # 从surface_path提取目录路径
        surface_path = config.get('surface_path', '')
        if not surface_path:
            print(f"  配置项 {i}: 缺少surface_path")
            continue
            
        # 构建num_parts.json的路径
        # 例如: data/preprocessed_data_messy_kitchen/xxx_combined/points.npy
        # 对应的num_parts.json在: data/preprocessed_data_messy_kitchen/xxx_combined/num_parts.json
        dir_path = os.path.dirname(surface_path)
        num_parts_file = os.path.join(dir_path, 'num_parts.json')
        
        try:
            if os.path.exists(num_parts_file):
                with open(num_parts_file, 'r', encoding='utf-8') as f:
                    num_parts_data = json.load(f)
                    new_num_parts = num_parts_data.get('num_parts')
                    
                    if new_num_parts is not None:
                        old_num_parts = config.get('num_parts')
                        config['num_parts'] = new_num_parts
                        updated_count += 1
                        print(f"  配置项 {i}: {old_num_parts} -> {new_num_parts}")
                    else:
                        print(f"  配置项 {i}: num_parts.json中缺少num_parts字段")
                        error_count += 1
            else:
                print(f"  配置项 {i}: 找不到文件 {num_parts_file}")
                error_count += 1
                
        except Exception as e:
            print(f"  配置项 {i}: 读取{num_parts_file}时出错: {e}")
            error_count += 1
    
    # 保存更新后的配置文件
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)
    
    print(f"完成! 更新了 {updated_count} 项，错误 {error_count} 项")
    print(f"配置文件已保存: {config_file_path}")
    return updated_count, error_count

def main():
    """主函数"""
    # 两个配置文件的路径
    config_files = [
        "data/preprocessed_data_messy_kitchen/messy_kitchen_configs.json",
        "data/preprocessed_data_messy_kitchen_scenes_part2/messy_kitchen_configs.json"
    ]
    
    total_updated = 0
    total_errors = 0
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"\n{'='*60}")
            updated, errors = update_config_file(config_file)
            total_updated += updated
            total_errors += errors
        else:
            print(f"\n配置文件不存在: {config_file}")
    
    print(f"\n{'='*60}")
    print(f"总计: 更新了 {total_updated} 项，错误 {total_errors} 项")

if __name__ == "__main__":
    main()
