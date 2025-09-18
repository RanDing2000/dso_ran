# PartCrafter评估脚本使用说明

本目录包含了用于评估PartCrafter模型性能的大规模测试脚本。

## 文件说明

- `eval.py`: 主要的评估脚本，包含完整的评估流程
- `run_eval.sh`: 便捷的shell脚本，用于运行评估
- `README_eval.md`: 本说明文档

## 功能特性

### 评估指标
- **Chamfer Distance**: 衡量生成网格与Ground Truth的几何相似性
- **F-Score**: 基于距离阈值的精确度和召回率综合指标
- **IoU (Intersection over Union)**: 体素级别的重叠度指标
- **Scene IoU**: 场景级别的部分间重叠度指标

### GAPS工具支持
- **GAPS工具**: 支持GAPS工具进行网格对齐和评估
- **自动检测**: 自动检测GAPS是否已安装，如未安装则尝试构建
- **评估方法**: 参考SSR评估方法进行PartCrafter性能评估

### 输出内容
- 详细的评估结果JSON文件
- CSV格式的结果表格
- 可视化图表（分布图、散点图等）
- 对比渲染图像（输入图像 + GT + 预测结果）
- 生成的3D网格文件（GLB格式）
- 网格统计信息和摘要

## 使用方法

### 1. 使用Python脚本直接运行

```bash
# 基础评估（仅PartCrafter）
python scripts/eval.py \
    --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \
    --output_dir results/evaluation_messy_kitchen \
    --model_path pretrained_weights/PartCrafter-Scene \
    --device cuda \
    --num_samples 10000 \
    --seed 0

# 启用GAPS工具支持
python scripts/eval.py \
    --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \
    --output_dir results/evaluation_messy_kitchen \
    --build_gaps
```

### 2. 使用Shell脚本运行

```bash
# 使用默认参数（仅PartCrafter）
bash scripts/run_eval.sh

# 启用GAPS工具支持
bash scripts/run_eval.sh --build_gaps

# 自定义参数
bash scripts/run_eval.sh \
    --config_path your_config.json \
    --output_dir your_output_dir \
    --device cuda \
    --num_samples 5000 \
    --build_gaps
```

### 3. 查看帮助信息

```bash
# Python脚本帮助
python scripts/eval.py --help

# Shell脚本帮助
bash scripts/run_eval.sh --help
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config_path` | str | `data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json` | 测试配置文件路径 |
| `--output_dir` | str | `results/evaluation_messy_kitchen` | 输出目录 |
| `--model_path` | str | `pretrained_weights/PartCrafter-Scene` | 模型权重路径 |
| `--device` | str | `cuda` | 计算设备 |
| `--num_samples` | int | `10000` | 评估采样点数 |
| `--seed` | int | `0` | 随机种子 |
| `--build_gaps` | flag | `True` | 设置GAPS工具用于评估（默认：已安装） |

## 配置文件格式

配置文件应为JSON格式，包含测试案例列表：

```json
[
  {
    "mesh_path": "datasets/messy_kitchen_test/example.glb",
    "surface_path": "data/preprocessed_data/example/points.npy",
    "image_path": "data/preprocessed_data/example/rendering_rmbg.png",
    "num_parts": 6,
    "iou_mean": 0.00037,
    "iou_max": 0.00173,
    "valid": true
  }
]
```

## 输出结构

评估完成后，输出目录将包含以下文件：

```
results/evaluation_messy_kitchen/
├── evaluation_results.json          # 详细评估结果
├── evaluation_results.csv           # 结果表格
├── evaluation_visualization.png     # 可视化图表
├── case_name_comparison.png         # 对比渲染图像
├── case_name/                       # 每个案例的网格文件
│   ├── part_00.glb                  # 生成的各部分网格
│   ├── part_01.glb
│   ├── merged.glb                   # 合并后的网格
│   ├── gt_part_00.glb              # Ground Truth各部分网格
│   ├── gt_part_01.glb
│   └── gt.glb                      # Ground Truth合并网格
└── mesh_summary.json               # 网格统计信息摘要
```

## 评估结果解读

### 指标含义
- **Chamfer Distance**: 越小越好，表示几何相似性越高
- **F-Score**: 0-1之间，越大越好，表示精确度和召回率的平衡
- **IoU**: 0-1之间，越大越好，表示体素重叠度
- **Scene IoU**: 0-1之间，越小越好，表示部分间分离度

### 可视化图表
- **分布图**: 显示各指标的分布情况
- **散点图**: 显示部分数量与性能的关系
- **对比图像**: 直观比较输入、GT和预测结果

## 注意事项

1. **硬件要求**: 需要CUDA GPU，建议至少8GB显存
2. **依赖安装**: 确保已安装所有必要的Python包
3. **模型下载**: 首次运行会自动下载PartCrafter-Scene模型
4. **内存使用**: 大量案例评估时注意内存使用情况
5. **时间消耗**: 完整评估可能需要较长时间，建议分批进行
6. **GAPS工具**: 评估使用GAPS工具进行网格对齐，默认假设已安装
7. **GAPS构建**: 如GAPS未安装，脚本会尝试自动构建（可能需要sudo权限）

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`num_samples`参数
   - 使用更小的模型或减少推理步数

2. **模型下载失败**
   - 检查网络连接
   - 手动下载模型到指定路径

3. **评估指标计算错误**
   - 检查GT网格文件是否完整
   - 验证网格格式是否正确

4. **渲染失败**
   - 确保OpenGL环境正确配置
   - 检查渲染依赖库

5. **GAPS构建失败**
   - 检查系统是否安装了必要的编译工具
   - 确保有足够的磁盘空间和权限
   - 手动运行`cd submodules/SSR-code/external && sudo bash build_gaps.sh`
   - 或者跳过GAPS构建，使用`--no_build_gaps`参数

6. **GAPS工具不可用**
   - 检查GAPS是否已正确安装
   - 验证SSR代码库路径是否正确
   - 确保GAPS工具可执行权限

### 调试模式

可以修改脚本中的参数进行调试：
- 减少测试案例数量
- 降低采样点数
- 使用CPU模式测试

## 扩展功能

脚本设计为模块化，可以轻松扩展：
- 添加新的评估指标
- 支持不同的数据集格式
- 集成其他3D生成模型
- 添加更多可视化选项
