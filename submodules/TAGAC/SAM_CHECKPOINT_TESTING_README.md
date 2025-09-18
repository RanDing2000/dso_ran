# SAM Checkpoint Testing with TARGO

This document describes how to use the enhanced `inference_vgn_targo_sam.py` script to test multiple SAM (Segment Anything Model) checkpoints with TARGO.

## Features

### Enhanced Robustness
- **Error Handling**: Comprehensive error handling for SAM initialization failures, model loading errors, and runtime exceptions
- **Fallback Mechanisms**: Graceful fallback when SAM segmentation fails
- **Progress Tracking**: Detailed progress reporting for each experiment
- **Result Validation**: Automatic validation of results and error reporting

### Multiple Checkpoint Support
- **Batch Testing**: Test multiple SAM checkpoints automatically
- **Comparison Reports**: Generate detailed comparison reports across different checkpoints
- **Organized Results**: Each checkpoint gets its own result directory
- **Flexible Input**: Support for different checkpoint file formats (.pth, .pt, .ckpt)

## Usage

### 1. Single SAM Checkpoint Testing

```bash
python scripts/inference/inference_vgn_targo_sam.py \
    --model_type targo \
    --occlusion-level slight \
    --max_scenes 10 \
    --vis true \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth
```

### 2. Multiple SAM Checkpoint Testing

```bash
python scripts/inference/inference_vgn_targo_sam.py \
    --model_type targo \
    --occlusion-level slight \
    --max_scenes 10 \
    --vis true \
    --sam_checkpoint_dir checkpoints/sam_checkpoints/
```

### 3. Create Test Script

```bash
python scripts/inference/inference_vgn_targo_sam.py --create-test-script
```

This creates a `test_sam_checkpoints.sh` script that you can customize and run.

### 4. Run Test Script

```bash
./test_sam_checkpoints.sh
```

## Command Line Arguments

### SAM Configuration
- `--sam_model_type`: SAM model type (vit_h, vit_l, vit_b)
- `--sam_checkpoint`: Path to single SAM checkpoint file
- `--sam_checkpoint_dir`: Directory containing multiple SAM checkpoints (overrides --sam_checkpoint)

### Model Configuration
- `--model_type`: TARGO model type (targo, targo_partial, targo_full_gt)
- `--model`: Path to TARGO model checkpoint
- `--shape_completion`: Whether to use shape completion
- `--sc_model_path`: Path to shape completion model

### Dataset Configuration
- `--occlusion-level`: Occlusion level (no, slight, medium)
- `--test_root`: Root directory of test dataset
- `--max_scenes`: Maximum number of scenes to process (0 for all)

### Visualization
- `--vis`: Enable visualization
- `--vis_failure_only`: Visualize only failure cases
- `--vis_rgb`: Visualize RGB images
- `--vis_gt_target`: Visualize ground truth target mesh
- `--vis_grasps`: Visualize grasps on affordance mesh
- `--vis_target_mesh_pred`: Save predicted target mesh

## Directory Structure

When testing multiple checkpoints, the script creates the following structure:

```
targo_eval_results/vgn/eval_results/targo/targo_sam/
├── 2025-01-XX_XX-XX-XX_sam_vit_h_4b8939/
│   ├── initial.txt
│   ├── occ_level_sr.json
│   ├── meta_evaluations.txt
│   ├── sam_segmentation_results.txt
│   ├── sam_segmentation_summary.json
│   ├── filtered_result.csv
│   └── result_summary.txt
├── 2025-01-XX_XX-XX-XX_sam_vit_l_4b8939/
│   └── ...
├── 2025-01-XX_XX-XX-XX_sam_vit_b_4b8939/
│   └── ...
└── comparison_results.json
```

## Output Files

### Individual Experiment Results
- `initial.txt`: Configuration parameters used
- `occ_level_sr.json`: Overall success rate
- `meta_evaluations.txt`: Detailed evaluation results
- `sam_segmentation_results.txt`: SAM segmentation metrics
- `sam_segmentation_summary.json`: SAM segmentation summary
- `filtered_result.csv`: Filtered results by bins
- `result_summary.txt`: Text summary of results

### Comparison Results
- `comparison_results.json`: Comparison across all checkpoints

## Error Handling

The script includes comprehensive error handling:

1. **SAM Initialization Errors**: Falls back to ground truth segmentation
2. **Model Loading Errors**: Reports specific error messages
3. **Runtime Errors**: Continues with next checkpoint
4. **File I/O Errors**: Graceful handling of missing files
5. **Validation Errors**: Clear error messages for configuration issues

## Example Results

### SAM Segmentation Metrics
```
=== SAM SEGMENTATION RESULTS ===
Total scenes: 100
Successful segmentations: 85
Segmentation success rate: 0.8500 (85.00%)
Average IoU: 0.7234
Average Dice: 0.8391
===================================
```

### Comparison Report
```
SAM CHECKPOINT COMPARISON REPORT
================================================================================
Checkpoint: sam_vit_h_4b8939
  Success Rate: 0.8500
  Result Path: /path/to/results/2025-01-XX_XX-XX-XX_sam_vit_h_4b8939
  Segmentation Success Rate: 0.8500
  Average IoU: 0.7234
  Average Dice: 0.8391

Checkpoint: sam_vit_l_4b8939
  Success Rate: 0.8200
  Result Path: /path/to/results/2025-01-XX_XX-XX-XX_sam_vit_l_4b8939
  Segmentation Success Rate: 0.8200
  Average IoU: 0.7102
  Average Dice: 0.8298
```

## Troubleshooting

### Common Issues

1. **"No SAM checkpoint files found"**
   - Check that the directory contains files with extensions: .pth, .pt, .ckpt
   - Verify the directory path is correct

2. **"Failed to run target_sample_offline_vgn.run()"**
   - This might be due to the generate_targo_input_data return value mismatch
   - The script will continue with error handling and use fallback values

3. **"SAM checkpoint not found"**
   - Verify the checkpoint file exists
   - Check file permissions
   - Download SAM checkpoints from: https://github.com/facebookresearch/segment-anything#model-checkpoints

4. **"Could not save results to JSON"**
   - Check disk space
   - Verify write permissions to result directory

### Performance Tips

1. **Reduce max_scenes** for faster testing (e.g., --max_scenes 10)
2. **Disable visualization** for faster execution (--vis false)
3. **Use smaller SAM models** (vit_b instead of vit_h) for faster inference
4. **Test on subset of data** first before running full evaluation

## Advanced Usage

### Custom Checkpoint Directory Structure
```
checkpoints/sam_checkpoints/
├── sam_vit_h_4b8939.pth
├── sam_vit_l_4b8939.pth
├── sam_vit_b_4b8939.pth
├── custom_sam_model.pth
└── fine_tuned_sam.pth
```

### Batch Testing Script
```bash
#!/bin/bash
# Custom batch testing script

SAM_DIRS=(
    "checkpoints/sam_vit_h"
    "checkpoints/sam_vit_l"
    "checkpoints/sam_vit_b"
    "checkpoints/custom_sam"
)

for sam_dir in "${SAM_DIRS[@]}"; do
    echo "Testing: $sam_dir"
    python scripts/inference/inference_vgn_targo_sam.py \
        --model_type targo \
        --occlusion-level slight \
        --max_scenes 50 \
        --vis false \
        --sam_checkpoint_dir "$sam_dir"
done
```

## Integration with Existing Workflows

The enhanced script is fully compatible with existing TARGO workflows:

1. **Same command line interface** for single checkpoint testing
2. **Backward compatible** with existing result formats
3. **Extends functionality** without breaking changes
4. **Maintains all existing features** while adding new capabilities

## Future Enhancements

Potential future improvements:

1. **Parallel processing** of multiple checkpoints
2. **Automatic checkpoint discovery** from model repositories
3. **Integration with experiment tracking** (Weights & Biases, MLflow)
4. **Automated hyperparameter optimization** for SAM models
5. **Real-time monitoring** of experiment progress

