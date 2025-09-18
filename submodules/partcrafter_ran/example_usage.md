# PartCrafter 评估脚本使用示例

## 功能说明

修改后的评估脚本支持使用现有预测结果进行评估，避免重复运行推理，但仍会进行对齐和计算指标。

## 主要功能

1. **自动检测现有结果**: 检查是否存在 `gt_merged.glb` 和 `pred_merged.glb` 等文件
2. **跳过推理**: 如果存在现有结果且未强制推理，则跳过推理步骤
3. **保持评估**: 即使使用现有结果，仍会进行对齐和计算指标
4. **灵活控制**: 通过命令行参数控制行为

## 命令行参数

```bash
python scripts/eval.py [options]
```

### 主要参数

- `--use_existing_results`: 使用现有预测结果（默认: True）
- `--force_inference`: 强制运行推理，覆盖现有结果（默认: False）
- `--config_path`: 测试配置文件路径
- `--output_dir`: 输出目录
- `--num_samples`: 评估采样点数（默认: 10000）

## 使用示例

### 1. 使用现有结果进行评估（推荐）

```bash
python scripts/eval.py \
    --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \
    --output_dir results/evaluation_messy_kitchen \
    --use_existing_results \
    --num_samples 10000
```

**行为**:
- 检查每个案例是否存在现有结果
- 如果存在，跳过推理，直接使用现有预测结果
- 仍然进行对齐和计算指标
- 如果不存在，运行推理生成新结果

### 2. 强制重新运行推理

```bash
python scripts/eval.py \
    --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \
    --output_dir results/evaluation_messy_kitchen \
    --force_inference \
    --num_samples 10000
```

**行为**:
- 忽略现有结果
- 对所有案例重新运行推理
- 覆盖现有预测结果

### 3. 禁用使用现有结果

```bash
python scripts/eval.py \
    --config_path data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json \
    --output_dir results/evaluation_messy_kitchen \
    --no-use_existing_results \
    --num_samples 10000
```

**行为**:
- 不使用现有结果
- 对所有案例运行推理
- 生成新的预测结果

## 文件结构

评估脚本会检查以下文件结构：

```
results/evaluation_messy_kitchen/
└── {case_name}/
    ├── gt_merged.glb          # 合并的GT网格
    ├── pred_merged.glb        # 合并的预测网格
    ├── gt_part_00.glb         # 单个GT部件
    ├── gt_part_01.glb
    ├── ...
    ├── pred_part_00.glb       # 单个预测部件
    ├── pred_part_01.glb
    ├── ...
    ├── mesh_summary.json      # 网格摘要信息
    └── {case_name}_comparison.png  # 对比图像
```

## 检测逻辑

脚本会按以下顺序检查现有结果：

1. 检查 `{case_name}/` 目录是否存在
2. 检查 `pred_part_*.glb` 文件（单个预测部件）
3. 检查 `gt_part_*.glb` 文件（单个GT部件）
4. 如果单个部件不存在，检查 `pred_merged.glb` 和 `gt_merged.glb`

## 输出信息

运行时会显示类似以下信息：

```
Evaluating case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed

Using existing results for case: bb7c492421494988a9abfd8e1accb0cd_combined_fixed
Loaded 6 predicted meshes and 6 GT meshes from existing results
Computing metrics with alignment for 6 predicted meshes...
```

## 注意事项

1. **模型加载**: 即使使用现有结果，仍需要加载模型（用于其他功能）
2. **内存使用**: 使用现有结果可以减少GPU内存使用
3. **时间节省**: 跳过推理可以显著节省评估时间
4. **结果一致性**: 确保现有结果的质量和完整性

## 故障排除

### 问题1: 找不到现有结果
- 检查文件路径是否正确
- 确认文件命名格式是否符合预期
- 验证文件权限

### 问题2: 现有结果损坏
- 使用 `--force_inference` 重新生成结果
- 检查磁盘空间是否充足

### 问题3: 对齐失败
- 检查GAPS工具是否正确安装
- 验证网格文件是否完整
- 查看错误日志获取详细信息
