#!/usr/bin/env python3
"""
Test script for integrating attention-map-diffusers with PartCrafter
"""

import os
import sys
import torch
from PIL import Image

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from huggingface_hub import snapshot_download

# Test attention-map-diffusers import
print("Testing attention-map-diffusers integration...")

try:
    from attention_map_diffusers import (
        attn_maps,
        init_pipeline,
        save_attention_maps
    )
    print("✓ Successfully imported attention-map-diffusers")
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"✗ Failed to import attention-map-diffusers: {e}")
    ATTENTION_AVAILABLE = False

if not ATTENTION_AVAILABLE:
    print("\nTo install attention-map-diffusers:")
    print("cd submodules/attention-map-diffusers")
    print("pip install -e .")
    sys.exit(1)

def test_partcrafter_architecture():
    """Test PartCrafter architecture to understand how to integrate attention visualization"""
    
    print("\n=== Testing PartCrafter Architecture ===")
    
    # Download and load model
    partcrafter_weights_dir = "pretrained_weights/PartCrafter-Scene"
    snapshot_download(repo_id="wgsxm/PartCrafter-Scene", local_dir=partcrafter_weights_dir)
    
    pipe = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to("cuda", torch.float16)
    
    print(f"Pipeline type: {type(pipe)}")
    print(f"Pipeline attributes: {dir(pipe)}")
    
    # Check for transformer
    if hasattr(pipe, 'transformer'):
        print(f"✓ Found transformer: {pipe.transformer.__class__.__name__}")
        print(f"Transformer type: {type(pipe.transformer)}")
        
        # Analyze transformer structure
        print("\nTransformer structure:")
        for name, module in pipe.transformer.named_modules():
            if 'attn' in name:
                print(f"  {name}: {type(module)}")
                if hasattr(module, 'processor'):
                    print(f"    - Has processor: {type(module.processor)}")
                if hasattr(module, 'is_cross_attention'):
                    print(f"    - is_cross_attention: {module.is_cross_attention}")
                if hasattr(module, 'cross_attention_dim'):
                    print(f"    - cross_attention_dim: {module.cross_attention_dim}")
    else:
        print("✗ No transformer found in pipeline")
    
    # Check for UNet (alternative)
    if hasattr(pipe, 'unet'):
        print(f"✓ Found UNet: {pipe.unet.__class__.__name__}")
    else:
        print("✗ No UNet found in pipeline")
    
    return pipe

def test_attention_hooks(pipe):
    """Test setting up attention hooks"""
    
    print("\n=== Testing Attention Hooks ===")
    
    # Clear any existing attention maps
    attn_maps.clear()
    
    # Try to initialize pipeline for attention visualization
    try:
        # For PartCrafter, we need to adapt the attention visualization
        # Let's try to manually set up hooks
        
        def hook_function(name, detach=True):
            def forward_hook(module, input, output):
                print(f"Hook triggered for {name}")
                if hasattr(module.processor, "attn_map"):
                    timestep = getattr(module.processor, "timestep", 0)
                    attn_maps[timestep] = attn_maps.get(timestep, dict())
                    attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
                    print(f"  Captured attention map for {name} at timestep {timestep}")
                    del module.processor.attn_map
            return forward_hook
        
        hooks = []
        
        # Register hooks for cross attention layers
        if hasattr(pipe, 'transformer'):
            for name, module in pipe.transformer.named_modules():
                if 'attn' in name and hasattr(module, 'processor'):
                    print(f"Registering hook for {name}")
                    if hasattr(module.processor, 'store_attn_map'):
                        module.processor.store_attn_map = True
                    hook = module.register_forward_hook(hook_function(name))
                    hooks.append(hook)
        
        print(f"Registered {len(hooks)} hooks")
        
        return hooks
        
    except Exception as e:
        print(f"✗ Failed to setup attention hooks: {e}")
        return []

def test_simple_inference(pipe, hooks):
    """Test simple inference to see if hooks work"""
    
    print("\n=== Testing Simple Inference ===")
    
    # Create a simple test image
    test_image = Image.new('RGB', (512, 512), color='white')
    
    try:
        # Run a simple forward pass
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
        
        # Check if attention maps were captured
        if attn_maps:
            print(f"✓ Captured attention maps for {len(attn_maps)} timesteps")
            for timestep, layers in attn_maps.items():
                print(f"  Timestep {timestep}: {len(layers)} layers")
        else:
            print("✗ No attention maps captured")
        
        return outputs
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return None

def main():
    print("PartCrafter + Attention-Map-Diffusers Integration Test")
    print("=" * 60)
    
    # Test architecture
    pipe = test_partcrafter_architecture()
    
    # Test attention hooks
    hooks = test_attention_hooks(pipe)
    
    # Test inference
    outputs = test_simple_inference(pipe, hooks)
    
    # Cleanup hooks
    for hook in hooks:
        hook.remove()
    
    print("\n=== Test Summary ===")
    if outputs is not None:
        print("✓ Basic integration test passed")
        print("✓ PartCrafter inference works")
        if attn_maps:
            print("✓ Attention maps captured")
            print("  You can now use the full inference script with --attention_viz")
        else:
            print("⚠ Attention maps not captured - may need architecture-specific adaptation")
    else:
        print("✗ Integration test failed")
    
    print("\nNext steps:")
    print("1. If attention maps were captured, try the full inference script:")
    print("   python scripts/inference_partcrafter_with_attention.py --image_path <image> --num_parts 3 --attention_viz")
    print("2. If not, the architecture may need specific adaptation for PartCrafter")

if __name__ == "__main__":
    main()
