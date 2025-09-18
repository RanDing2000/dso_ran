#!/bin/bash

# PartCrafter大规模评估脚本
# 用于评估PartCrafter在messy kitchen数据集上的性能

echo "=== PartCrafter大规模评估脚本 ==="
echo "开始时间: $(date)"

# 设置默认参数
CONFIG_PATH="data/preprocessed_data_messy_kitchen_scenes_test/messy_kitchen_configs.json"
OUTPUT_DIR="results/evaluation_messy_kitchen"
MODEL_PATH="pretrained_weights/PartCrafter-Scene"
DEVICE="cuda"
NUM_SAMPLES=10000
SEED=0
BUILD_GAPS=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --build_gaps)
            BUILD_GAPS=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --config_path PATH    测试配置文件路径 (默认: $CONFIG_PATH)"
            echo "  --output_dir PATH     输出目录 (默认: $OUTPUT_DIR)"
            echo "  --model_path PATH     模型权重路径 (默认: $MODEL_PATH)"
            echo "  --device DEVICE       计算设备 (默认: $DEVICE)"
            echo "  --num_samples N       评估采样点数 (默认: $NUM_SAMPLES)"
            echo "  --seed SEED           随机种子 (默认: $SEED)"
            echo "  --build_gaps          设置GAPS工具用于评估 (默认: 已安装)"
            echo "  -h, --help            显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 -h 或 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

echo "配置参数:"
echo "  配置文件: $CONFIG_PATH"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型路径: $MODEL_PATH"
echo "  计算设备: $DEVICE"
echo "  采样点数: $NUM_SAMPLES"
echo "  随机种子: $SEED"
echo "  构建GAPS: $BUILD_GAPS"
echo ""

# 运行评估脚本
echo "开始运行评估..."
python scripts/eval.py \
    --config_path "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --device "$DEVICE" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --build_gaps

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "=== 评估完成 ==="
    echo "结束时间: $(date)"
    echo "结果保存在: $OUTPUT_DIR"
    echo ""
    echo "生成的文件:"
    ls -la "$OUTPUT_DIR"
else
    echo ""
    echo "=== 评估失败 ==="
    echo "请检查错误信息并重试"
    exit 1
fi
