import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from PIL import Image
import trimesh
from einops import rearrange

class AttentionVisualizer:
    """
    A utility class for visualizing cross-attention weights in PartCrafter model.
    """
    
    def __init__(self):
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self, model):
        """
        Register hooks to capture attention weights from the model.
        
        Args:
            model: The PartCrafter model
        """
        self.clear_hooks()
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # For cross attention, we want to capture the attention weights
                if hasattr(module, 'is_cross_attention') and module.is_cross_attention:
                    # The attention weights are computed in the processor
                    # We need to capture them during the forward pass
                    self.attention_maps[name] = {
                        'module': module,
                        'input': input,
                        'output': output
                    }
            return hook
        
        # Register hooks for cross attention layers
        cross_attention_count = 0
        for name, module in model.named_modules():
            # Check if this is a cross attention layer
            is_cross_attn = False
            
            # Method 1: Check is_cross_attention attribute
            if hasattr(module, 'is_cross_attention') and module.is_cross_attention:
                is_cross_attn = True
            
            # Method 2: Check if it's attn2 (cross attention in DiT blocks)
            elif 'attn2' in name and hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                is_cross_attn = True
            
            # Method 3: Check if it has cross_attention_dim
            elif hasattr(module, 'cross_attention_dim') and module.cross_attention_dim is not None:
                is_cross_attn = True
            
            if is_cross_attn:
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
                cross_attention_count += 1
        
        if cross_attention_count == 0:
            print("WARNING: No cross attention layers found! This might be the issue.")
            # Let's print all attention layers for debugging
            print("DEBUG: Available attention layers:")
            for name, module in model.named_modules():
                if 'attn' in name and hasattr(module, 'to_q'):
                    print(f"  - {name}: {type(module)}")
                
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()
    
    def extract_attention_weights(self, model, hidden_states, encoder_hidden_states, 
                                layer_idx: Optional[int] = None, attention_kwargs: Optional[Dict] = None,
                                timestep: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights from the model.
        
        Args:
            model: The PartCrafter model
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states for cross attention
            layer_idx: Specific layer to extract from (if None, extract from all)
            
        Returns:
            Dictionary containing attention weights for each layer
        """
        attention_weights = {}
        
        # Temporarily modify the attention processor to capture weights
        original_processors = {}
        
        def create_capturing_processor(original_processor, layer_name):
            def capturing_processor(attn, hidden_states, encoder_hidden_states=None, **kwargs):
                # Get query, key, value
                query = attn.to_q(hidden_states)
                key = attn.to_k(encoder_hidden_states) if encoder_hidden_states is not None else attn.to_k(hidden_states)
                value = attn.to_v(encoder_hidden_states) if encoder_hidden_states is not None else attn.to_v(hidden_states)
                
                # Reshape for attention computation
                batch_size = query.shape[0]
                head_dim = query.shape[-1] // attn.heads
                
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                # Apply normalization if exists
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                
                # Compute attention weights
                attention_scores = torch.matmul(query, key.transpose(-1, -2))
                attention_scores = attention_scores / (head_dim ** 0.5)
                attention_weights_raw = F.softmax(attention_scores, dim=-1)
                
                # Store the attention weights
                attention_weights[layer_name] = attention_weights_raw.detach()
                
                # Continue with normal forward pass - extract all necessary parameters from kwargs
                attention_mask = kwargs.get('attention_mask', None)
                temb = kwargs.get('temb', None)
                image_rotary_emb = kwargs.get('image_rotary_emb', None)
                num_parts = kwargs.get('num_parts', None)
                
                return original_processor(
                    attn, 
                    hidden_states, 
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    num_parts=num_parts
                )
            
            return capturing_processor
        
        # Replace processors temporarily
        for name, module in model.named_modules():
            if hasattr(module, 'is_cross_attention') and module.is_cross_attention:
                if layer_idx is None or f"layer_{layer_idx}" in name:
                    original_processors[name] = module.processor
                    module.processor = create_capturing_processor(module.processor, name)
        
        # Run forward pass
        with torch.no_grad():
            # Ensure timestep is provided; if not, create a zero tensor on the right device
            if timestep is None:
                batch_size = hidden_states.shape[0]
                device = hidden_states.device
                timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
            # Pass attention_kwargs if provided
            if attention_kwargs is None:
                attention_kwargs = {}
            # Ensure num_parts is in attention_kwargs if not already present
            if "num_parts" not in attention_kwargs:
                attention_kwargs["num_parts"] = hidden_states.shape[0]  # Use batch size as num_parts
            _ = model(hidden_states, timestep, encoder_hidden_states=encoder_hidden_states, attention_kwargs=attention_kwargs)
        
        # Restore original processors
        for name, processor in original_processors.items():
            for module_name, module in model.named_modules():
                if module_name == name:
                    module.processor = processor
                    break
        
        return attention_weights
    
    def visualize_attention_heatmap(self, attention_weights: torch.Tensor, 
                                  save_path: str, 
                                  title: str = "Cross Attention Heatmap",
                                  figsize: Tuple[int, int] = (12, 8)):
        """
        Create a heatmap visualization of attention weights.
        
        Args:
            attention_weights: Attention weights tensor of shape (batch, heads, seq_len, seq_len)
            save_path: Path to save the visualization
            title: Title for the plot
            figsize: Figure size
        """
        # Average over heads and batch
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
        elif attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=0)  # Average over heads
        
        # Convert to numpy
        attention_np = attention_weights.cpu().numpy()
        
        # Create the plot
        plt.figure(figsize=figsize)
        sns.heatmap(attention_np, 
                   cmap='viridis', 
                   annot=False, 
                   cbar=True,
                   square=True)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Key Sequence Position', fontsize=12)
        plt.ylabel('Query Sequence Position', fontsize=12)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def visualize_multi_head_attention(self, attention_weights: torch.Tensor,
                                     save_path: str,
                                     title: str = "Multi-Head Cross Attention",
                                     figsize: Tuple[int, int] = (20, 16)):
        """
        Visualize attention weights for all heads separately.
        
        Args:
            attention_weights: Attention weights tensor of shape (batch, heads, seq_len, seq_len)
            save_path: Path to save the visualization
            title: Title for the plot
            figsize: Figure size
        """
        # Average over batch dimension
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=0)  # Average over batch
        
        num_heads = attention_weights.shape[0]
        attention_np = attention_weights.cpu().numpy()
        
        # Calculate grid dimensions
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_heads):
            row = i // cols
            col = i % cols
            
            sns.heatmap(attention_np[i], 
                       cmap='viridis', 
                       annot=False, 
                       cbar=True,
                       square=True,
                       ax=axes[row, col])
            
            axes[row, col].set_title(f'Head {i+1}', fontsize=10)
            axes[row, col].set_xlabel('Key Position')
            axes[row, col].set_ylabel('Query Position')
        
        # Hide empty subplots
        for i in range(num_heads, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_part_attention(self, attention_weights: torch.Tensor,
                               num_parts: int,
                               save_path: str,
                               title: str = "Part-wise Cross Attention"):
        """
        Visualize attention weights specifically for part-based attention.
        
        Args:
            attention_weights: Attention weights tensor
            num_parts: Number of parts
            save_path: Path to save the visualization
            title: Title for the plot
        """
        # For part-based attention, we might want to see how each part attends to different regions
        attention_np = attention_weights.mean(dim=(0, 1)).cpu().numpy()  # Average over batch and heads
        
        # Reshape if needed for part-based visualization
        seq_len = attention_np.shape[0]
        tokens_per_part = seq_len // num_parts
        
        fig, axes = plt.subplots(1, num_parts, figsize=(4*num_parts, 4))
        if num_parts == 1:
            axes = [axes]
        
        for part_idx in range(num_parts):
            start_idx = part_idx * tokens_per_part
            end_idx = start_idx + tokens_per_part
            
            part_attention = attention_np[start_idx:end_idx, :]
            
            sns.heatmap(part_attention, 
                       cmap='viridis', 
                       annot=False, 
                       cbar=True,
                       ax=axes[part_idx])
            
            axes[part_idx].set_title(f'Part {part_idx+1}', fontsize=12)
            axes[part_idx].set_xlabel('Key Position')
            axes[part_idx].set_ylabel('Query Position (Part Tokens)')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_animation(self, attention_weights_sequence: List[torch.Tensor],
                                 save_path: str,
                                 fps: int = 2):
        """
        Create an animation showing attention weights over multiple timesteps.
        
        Args:
            attention_weights_sequence: List of attention weights for different timesteps
            save_path: Path to save the animation
            fps: Frames per second
        """
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def animate(frame):
            ax.clear()
            attention_np = attention_weights_sequence[frame].mean(dim=(0, 1)).cpu().numpy()
            sns.heatmap(attention_np, cmap='viridis', annot=False, cbar=True, square=True, ax=ax)
            ax.set_title(f'Cross Attention - Timestep {frame}', fontsize=14)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(attention_weights_sequence), 
                                     interval=1000//fps, repeat=True)
        
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=fps)
        plt.close()

def visualize_cross_attention_during_inference(model, pipeline, image, 
                                             output_dir: str = "attention_visualizations",
                                             num_inference_steps: int = 20):
    """
    Visualize cross attention during the inference process.
    
    Args:
        model: The PartCrafter model
        pipeline: The inference pipeline
        image: Input image
        output_dir: Directory to save visualizations
        num_inference_steps: Number of inference steps to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = AttentionVisualizer()
    
    # Register hooks to capture attention weights
    visualizer.register_hooks(model)
    
    # Run inference with attention capture
    attention_weights_sequence = []
    
    def callback_fn(step, timestep, latents):
        # Extract attention weights at this step
        if hasattr(model, 'attention_maps'):
            step_attention = {}
            for name, attention_data in model.attention_maps.items():
                step_attention[name] = attention_data
            attention_weights_sequence.append(step_attention)
    
    # Run inference
    with torch.no_grad():
        result = pipeline(
            image,
            num_inference_steps=num_inference_steps,
            callback=callback_fn,
            callback_steps=1
        )
    
    # Create visualizations
    for step, attention_weights in enumerate(attention_weights_sequence):
        for layer_name, weights in attention_weights.items():
            # Create heatmap
            heatmap_path = os.path.join(output_dir, f"step_{step:03d}_{layer_name}_heatmap.png")
            visualizer.visualize_attention_heatmap(
                weights, 
                heatmap_path, 
                title=f"Cross Attention - Step {step}, {layer_name}"
            )
            
            # Create multi-head visualization
            multihead_path = os.path.join(output_dir, f"step_{step:03d}_{layer_name}_multihead.png")
            visualizer.visualize_multi_head_attention(
                weights,
                multihead_path,
                title=f"Multi-Head Cross Attention - Step {step}, {layer_name}"
            )
    
    # Create animation
    if len(attention_weights_sequence) > 1:
        animation_path = os.path.join(output_dir, "attention_animation.gif")
        # Extract first layer attention weights for animation
        first_layer_weights = [list(step.values())[0] for step in attention_weights_sequence]
        visualizer.create_attention_animation(first_layer_weights, animation_path)
    
    # Clean up
    visualizer.clear_hooks()
    
    print(f"Attention visualizations saved to {output_dir}")
    return attention_weights_sequence
