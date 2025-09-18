# PartCrafter + Attention-Map-Diffusers 集成说明

本文档说明如何将 `attention-map-diffusers` 库与 PartCrafter 模型集成，以可视化交叉注意力图。

## 概述

`attention-map-diffusers` 是一个用于提取和可视化扩散模型交叉注意力图的工具库。我们尝试将其与 PartCrafter 模型集成，以分析模型在生成 3D 网格时如何关注输入图像的不同部分。

## 安装

### 1. 安装 attention-map-diffusers

```bash
cd submodules/attention-map-diffusers
pip install -e .
```

### 2. 验证安装

```bash
python scripts/test_attention_integration.py
```

## 使用方法

### 1. 测试集成

首先运行测试脚本，检查集成是否正常工作：

```bash
python scripts/test_attention_integration.py
```

这个脚本会：
- 检查 `attention-map-diffusers` 是否正确安装
- 分析 PartCrafter 的架构
- 测试注意力钩子的设置
- 运行简单的推理测试

### 2. 使用带注意力可视化的推理

```bash
python scripts/inference_partcrafter_with_attention.py \
    --image_path path/to/your/image.jpg \
    --num_parts 3 \
    --attention_viz \
    --output_dir ./results \
    --render
```

### 3. 参数说明

- `--image_path`: 输入图像路径
- `--num_parts`: 要生成的部件数量 (1-8)
- `--attention_viz`: 启用注意力可视化
- `--attention_viz_dir`: 注意力可视化输出目录（可选）
- `--output_dir`: 输出目录
- `--render`: 生成渲染图像
- `--seed`: 随机种子
- `--num_inference_steps`: 推理步数
- `--guidance_scale`: 引导尺度

## 输出结构

启用注意力可视化后，输出目录结构如下：

```
results/
└── 20241201_14_30_25/
    ├── part_00.glb          # 生成的部件网格
    ├── part_01.glb
    ├── part_02.glb
    ├── object.glb           # 合并的网格
    ├── rendering.png        # 渲染图像
    ├── rendering.gif        # 渲染动画
    └── attention_maps/      # 注意力图
        ├── timestep_0/
        │   └── layer_attn_0/
        │       ├── attention_weights.npy
        │       └── attention_heatmap.png
        ├── timestep_1/
        │   └── ...
        └── ...
```

## 注意力图解释

### 1. 注意力权重文件 (.npy)
- 包含原始的注意力权重张量
- 形状通常为 `(batch_size, num_heads, query_length, key_length)`
- 可以使用 NumPy 加载进行进一步分析

### 2. 注意力热力图 (.png)
- 可视化注意力权重的热力图
- 显示查询位置对键位置的关注程度
- 颜色越亮表示注意力权重越高

## 技术细节

### PartCrafter 架构适配

PartCrafter 使用基于 Transformer 的架构，类似于 Stable Diffusion 3。集成过程包括：

1. **架构分析**: 识别模型中的交叉注意力层
2. **钩子注册**: 在注意力层注册前向钩子
3. **权重捕获**: 在推理过程中捕获注意力权重
4. **可视化**: 将权重转换为热力图

### 限制和注意事项

1. **架构兼容性**: 由于 PartCrafter 的独特架构，可能需要特定的适配
2. **内存使用**: 注意力可视化会增加内存使用量
3. **性能影响**: 钩子注册可能略微影响推理速度
4. **实验性功能**: 这是一个实验性功能，可能需要根据具体模型版本进行调整

## 故障排除

### 1. 导入错误

如果遇到导入错误：

```bash
# 确保在正确的环境中
conda activate partcrafter

# 重新安装 attention-map-diffusers
cd submodules/attention-map-diffusers
pip install -e .
```

### 2. 没有捕获到注意力图

可能的原因：
- PartCrafter 架构与预期不同
- 注意力处理器类型不兼容
- 钩子注册失败

解决方案：
- 运行测试脚本分析架构
- 检查模型版本兼容性
- 查看错误日志

### 3. 内存不足

如果遇到内存不足：

```bash
# 减少推理步数
--num_inference_steps 20

# 减少 token 数量
--num_tokens 1024

# 使用更少的部件
--num_parts 2
```

## 示例

### 基本使用

```python
# 在 Python 中使用
from scripts.inference_partcrafter_with_attention import run_partcrafter_inference

# 运行推理
outputs, image = run_partcrafter_inference(
    pipe, 
    image_path="input.jpg", 
    num_parts=3
)
```

### 自定义注意力可视化

```python
# 自定义注意力图保存
def custom_attention_saver(attn_maps, output_dir):
    for timestep, layers in attn_maps.items():
        for layer_name, attn_map in layers.items():
            # 自定义处理逻辑
            pass
```

## 贡献

如果您在使用过程中发现问题或有改进建议，请：

1. 运行测试脚本并记录输出
2. 提供详细的错误信息
3. 说明您的环境和配置

## 参考资料

- [attention-map-diffusers GitHub](https://github.com/wooyeolBaek/attention-map-diffusers)
- [PartCrafter 论文](https://arxiv.org/abs/...)
- [Diffusers 文档](https://huggingface.co/docs/diffusers)
