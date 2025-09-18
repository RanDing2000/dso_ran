#!/usr/bin/env python3
import json
import random

def sample_configs(input_file, output_file, sample_size=100):
    """从配置文件中随机采样指定数量的配置项"""
    print(f"正在读取配置文件: {input_file}")
    
    # 读取原始配置文件
    with open(input_file, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    total_configs = len(configs)
    print(f"总配置项数量: {total_configs}")
    
    if sample_size > total_configs:
        print(f"警告: 采样数量 {sample_size} 大于总配置项数量 {total_configs}")
        sample_size = total_configs
    
    # 随机采样
    sampled_configs = random.sample(configs, sample_size)
    print(f"成功采样 {len(sampled_configs)} 个配置项")
    
    # 保存采样结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_configs, f, indent=2, ensure_ascii=False)
    
    print(f"采样结果已保存到: {output_file}")
    
    return sampled_configs

def main():
    """主函数"""
    input_file = "data/preprocessed_data_messy_kitchen_scenes_part2/messy_kitchen_configs.json"
    output_file = "data/preprocessed_data_messy_kitchen_scenes_part2/messy_kitchen_test_100.json"
    sample_size = 100
    
    # 设置随机种子以确保可重复性（可选）
    random.seed(42)
    
    try:
        sampled_configs = sample_configs(input_file, output_file, sample_size)
        print(f"\n采样完成! 共采样 {len(sampled_configs)} 个配置项")
        
        # 显示前几个采样的配置项作为示例
        print("\n前3个采样的配置项:")
        for i, config in enumerate(sampled_configs[:3]):
            print(f"  {i+1}. mesh_path: {config.get('mesh_path', 'N/A')}")
            print(f"     num_parts: {config.get('num_parts', 'N/A')}")
            print(f"     valid: {config.get('valid', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
