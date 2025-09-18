# Cross Attention Visualization Guide

This guide explains how to visualize and analyze cross-attention patterns in the PartCrafter model.

## Overview

Cross attention visualization helps understand how the model attends to different parts of the input during the generation process. This is particularly useful for:

- Understanding which parts of the input image influence each generated part
- Analyzing attention patterns across different layers
- Debugging model behavior
- Research and analysis

## Features

The `AttentionVisualizer` class provides:

1. **Attention Heatmaps**: Visualize attention weights as 2D heatmaps
2. **Multi-head Visualization**: Show attention patterns for each attention head
3. **Part-wise Analysis**: Analyze attention patterns specific to each part
4. **Temporal Analysis**: Track attention changes over inference steps
5. **Animation**: Create GIF animations of attention evolution

## Usage

### Basic Usage

```python
from src.utils.attention_visualization import AttentionVisualizer

# Create visualizer
visualizer = AttentionVisualizer()

# Register hooks to capture attention weights
visualizer.register_hooks(model)

# Run inference
with torch.no_grad():
    result = model(input_data)

# Extract attention weights
attention_weights = visualizer.extract_attention_weights(
    model, hidden_states, encoder_hidden_states
)

# Create visualizations
for layer_name, weights in attention_weights.items():
    # Heatmap
    visualizer.visualize_attention_heatmap(
        weights, 
        f"heatmap_{layer_name}.png",
        title=f"Cross Attention - {layer_name}"
    )
    
    # Multi-head visualization
    visualizer.visualize_multi_head_attention(
        weights,
        f"multihead_{layer_name}.png",
        title=f"Multi-Head - {layer_name}"
    )

# Clean up
visualizer.clear_hooks()
```

### During Inference

```python
def visualize_during_inference(pipeline, model, image, output_dir):
    visualizer = AttentionVisualizer()
    visualizer.register_hooks(model)
    
    # Run inference with callback
    attention_sequence = []
    
    def callback_fn(step, timestep, latents):
        # Capture attention at each step
        if hasattr(model, 'attention_maps'):
            attention_sequence.append(model.attention_maps.copy())
    
    result = pipeline(
        image,
        num_inference_steps=20,
        callback=callback_fn,
        callback_steps=1
    )
    
    # Create visualizations for each step
    for step, attention_weights in enumerate(attention_sequence):
        for layer_name, weights in attention_weights.items():
            visualizer.visualize_attention_heatmap(
                weights,
                f"{output_dir}/step_{step:03d}_{layer_name}.png"
            )
    
    visualizer.clear_hooks()
    return attention_sequence
```

## Visualization Types

### 1. Attention Heatmap

Shows attention weights as a 2D heatmap where:
- X-axis: Key sequence positions
- Y-axis: Query sequence positions
- Color intensity: Attention weight strength

```python
visualizer.visualize_attention_heatmap(
    attention_weights,
    "heatmap.png",
    title="Cross Attention Heatmap"
)
```

### 2. Multi-head Visualization

Shows attention patterns for each attention head separately:

```python
visualizer.visualize_multi_head_attention(
    attention_weights,
    "multihead.png",
    title="Multi-Head Cross Attention"
)
```

### 3. Part-wise Analysis

For part-based models, analyze attention patterns for each part:

```python
visualizer.visualize_part_attention(
    attention_weights,
    num_parts=8,
    "part_attention.png",
    title="Part-wise Cross Attention"
)
```

### 4. Temporal Animation

Create animations showing attention evolution over time:

```python
visualizer.create_attention_animation(
    attention_weights_sequence,
    "attention_animation.gif",
    fps=2
)
```

## Analysis Tips

### Understanding Attention Patterns

1. **High attention values** (bright colors) indicate strong connections
2. **Diagonal patterns** suggest self-attention or local attention
3. **Block patterns** may indicate part-based attention
4. **Sparse patterns** suggest focused attention on specific regions

### Common Patterns

- **Global attention**: Attention spread across the entire sequence
- **Local attention**: Attention focused on nearby positions
- **Part-specific attention**: Different parts attend to different regions
- **Hierarchical attention**: Attention patterns change across layers

### Debugging

- Check for **uniform attention** (may indicate training issues)
- Look for **attention collapse** (all attention on single position)
- Verify **part separation** in part-based models
- Monitor **attention stability** across timesteps

## Advanced Usage

### Custom Attention Analysis

```python
# Extract specific layer attention
layer_attention = visualizer.extract_attention_weights(
    model, hidden_states, encoder_hidden_states, layer_idx=5
)

# Analyze attention statistics
attention_stats = {
    'mean': attention_weights.mean().item(),
    'max': attention_weights.max().item(),
    'sparsity': (attention_weights < 0.01).float().mean().item(),
    'entropy': -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean().item()
}
```

### Batch Processing

```python
# Process multiple images
for image_path in image_paths:
    image = Image.open(image_path)
    attention_weights = visualize_during_inference(pipeline, model, image, output_dir)
    
    # Save results
    torch.save(attention_weights, f"attention_{image_path.stem}.pt")
```

## Troubleshooting

### Common Issues

1. **No attention weights captured**: Ensure hooks are properly registered
2. **Memory issues**: Use smaller batch sizes or fewer inference steps
3. **Empty visualizations**: Check if model has cross-attention layers
4. **Incorrect shapes**: Verify input tensor dimensions

### Performance Tips

- Use `torch.no_grad()` for inference
- Process attention weights on CPU for visualization
- Use smaller models for faster analysis
- Cache attention weights for repeated analysis

## Examples

See the `examples/` directory for complete working examples:

- `attention_visualization_example.py`: Basic usage example
- `part_attention_analysis.py`: Part-specific analysis
- `temporal_attention_analysis.py`: Temporal analysis

## Research Applications

This visualization tool is useful for:

- **Model interpretability**: Understanding model decisions
- **Architecture analysis**: Comparing different attention mechanisms
- **Training analysis**: Monitoring attention patterns during training
- **Part decomposition**: Analyzing how parts are learned
- **Cross-modal analysis**: Understanding image-to-part relationships
