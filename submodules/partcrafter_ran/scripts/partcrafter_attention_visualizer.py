#!/usr/bin/env python3
"""
PartCrafter-specific attention visualization tool
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from PIL import Image

class PartCrafterAttentionVisualizer:
    """
    Attention visualizer specifically designed for PartCrafter architecture
    """
    
    def __init__(self):
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self, transformer):
        """
        Register hooks to capture attention weights from PartCrafter transformer
        
        Args:
            transformer: PartCrafter transformer model
        """
        self.clear_hooks()
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # Capture attention weights during forward pass
                if hasattr(module, 'processor'):
                    # Try to access attention weights from processor
                    if hasattr(module.processor, 'attn_map'):
                        timestep = getattr(module.processor, 'timestep', 0)
                        self.attention_maps[timestep] = self.attention_maps.get(timestep, dict())
                        self.attention_maps[timestep][name] = module.processor.attn_map.detach().cpu()
                        print(f"  Captured attention map for {name} at timestep {timestep}")
                    else:
                        # Try to compute attention weights manually
                        try:
                            # Extract query, key, value from input
                            if len(input) >= 2:
                                hidden_states = input[0]
                                encoder_hidden_states = input[1] if len(input) > 1 else None
                                
                                # Compute attention weights manually
                                query = module.to_q(hidden_states)
                                key = module.to_k(encoder_hidden_states) if encoder_hidden_states is not None else module.to_k(hidden_states)
                                value = module.to_v(encoder_hidden_states) if encoder_hidden_states is not None else module.to_v(hidden_states)
                                
                                # Reshape for attention computation
                                batch_size = query.shape[0]
                                head_dim = query.shape[-1] // module.heads
                                
                                query = query.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
                                key = key.view(batch_size, -1, module.heads, head_dim).transpose(1, 2)
                                
                                # Compute attention scores
                                attention_scores = torch.matmul(query, key.transpose(-1, -2))
                                attention_scores = attention_scores / (head_dim ** 0.5)
                                attention_weights = F.softmax(attention_scores, dim=-1)
                                
                                # Store attention weights
                                timestep = 0  # Default timestep
                                self.attention_maps[timestep] = self.attention_maps.get(timestep, dict())
                                self.attention_maps[timestep][name] = attention_weights.detach().cpu()
                                print(f"  Manually captured attention map for {name} at timestep {timestep}")
                                
                        except Exception as e:
                            print(f"  Failed to capture attention for {name}: {e}")
                            
            return hook
        
        # Register hooks for cross attention layers (attn2)
        hook_count = 0
        for name, module in transformer.named_modules():
            if 'attn2' in name and hasattr(module, 'processor'):
                # Only capture cross attention (attn2)
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
                hook_count += 1
                print(f"Registered hook for {name}")
        
        print(f"Registered {hook_count} hooks for cross attention layers")
        
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_maps.clear()
    
    def save_attention_maps(self, output_dir: str, case_name: str = "partcrafter"):
        """
        Save captured attention maps
        
        Args:
            output_dir: Output directory
            case_name: Case name for file naming
        """
        if not self.attention_maps:
            print("No attention maps to save")
            return
        
        attention_dir = os.path.join(output_dir, "attention_maps")
        os.makedirs(attention_dir, exist_ok=True)
        
        print(f"Saving attention maps to: {attention_dir}")
        
        # Save attention maps as numpy arrays and visualizations
        for timestep, layers in self.attention_maps.items():
            timestep_dir = os.path.join(attention_dir, f"timestep_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)
            
            for layer_name, attn_map in layers.items():
                layer_dir = os.path.join(timestep_dir, f"layer_{layer_name}")
                os.makedirs(layer_dir, exist_ok=True)
                
                # Save as numpy array
                np_path = os.path.join(layer_dir, "attention_weights.npy")
                np.save(np_path, attn_map.numpy())
                
                # Create heatmap visualization
                try:
                    # Average over heads if multiple heads
                    if attn_map.dim() > 2:
                        avg_attn = attn_map.mean(dim=0)  # Average over heads
                    else:
                        avg_attn = attn_map
                    
                    # Convert to numpy
                    attn_np = avg_attn.numpy()
                    
                    # Create heatmap
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(attn_np, cmap='viridis', cbar=True)
                    plt.title(f"Cross Attention Heatmap - {layer_name}")
                    plt.xlabel("Key Position")
                    plt.ylabel("Query Position")
                    
                    # Save heatmap
                    heatmap_path = os.path.join(layer_dir, "attention_heatmap.png")
                    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Saved attention map for {layer_name}")
                    
                except Exception as e:
                    print(f"  Failed to create heatmap for {layer_name}: {e}")
    
    def get_attention_summary(self) -> Dict:
        """
        Get summary of captured attention maps
        
        Returns:
            Dictionary with attention map statistics
        """
        summary = {
            'total_timesteps': len(self.attention_maps),
            'total_layers': sum(len(layers) for layers in self.attention_maps.values()),
            'timesteps': list(self.attention_maps.keys()),
            'layers_per_timestep': {ts: len(layers) for ts, layers in self.attention_maps.items()}
        }
        
        if self.attention_maps:
            # Get shape information from first available map
            first_timestep = list(self.attention_maps.keys())[0]
            first_layer = list(self.attention_maps[first_timestep].keys())[0]
            first_map = self.attention_maps[first_timestep][first_layer]
            
            summary['attention_shape'] = list(first_map.shape)
            summary['num_heads'] = first_map.shape[0] if first_map.dim() > 2 else 1
        
        return summary


def test_partcrafter_attention():
    """Test PartCrafter attention visualization"""
    
    print("Testing PartCrafter attention visualization...")
    
    # Import PartCrafter
    import sys
    sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')
    
    from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from huggingface_hub import snapshot_download
    from PIL import Image
    
    # Download and load model
    partcrafter_weights_dir = "pretrained_weights/PartCrafter-Scene"
    snapshot_download(repo_id="wgsxm/PartCrafter-Scene", local_dir=partcrafter_weights_dir)
    
    pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to("cuda", torch.float16)
    
    # Create attention visualizer
    visualizer = PartCrafterAttentionVisualizer()
    
    # Register hooks
    visualizer.register_hooks(pipe.transformer)
    
    # Create test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    # Run inference
    print("Running test inference...")
    with torch.no_grad():
        outputs = pipe(
            image=[test_image] * 2,  # 2 parts
            attention_kwargs={"num_parts": 2},
            num_tokens=512,  # Reduced for testing
            generator=torch.Generator(device=pipe.device).manual_seed(0),
            num_inference_steps=5,  # Reduced for testing
            guidance_scale=7.0,
            max_num_expanded_coords=int(1e9),
            use_flash_decoder=False,
        ).meshes
    
    print(f"✓ Inference completed, generated {len(outputs)} meshes")
    
    # Get attention summary
    summary = visualizer.get_attention_summary()
    print(f"✓ Attention summary: {summary}")
    
    # Save attention maps
    output_dir = "./test_attention_output"
    visualizer.save_attention_maps(output_dir, "test_case")
    
    # Cleanup
    visualizer.clear_hooks()
    
    return summary


if __name__ == "__main__":
    test_partcrafter_attention()
