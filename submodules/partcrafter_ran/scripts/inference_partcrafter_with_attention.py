#!/usr/bin/env python3
"""
PartCrafter inference script with attention map visualization using attention-map-diffusers
"""

import argparse
import os
import sys
import time
from typing import Any, Union, List

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image
from accelerate.utils import set_seed

# Add project root to path
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

from src.utils.data_utils import get_colored_mesh_composition
from src.utils.render_utils import render_views_around_mesh, render_normal_views_around_mesh, make_grid_for_images_or_videos, export_renderings
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline

# Import attention-map-diffusers utilities
try:
    from attention_map_diffusers import (
        attn_maps,
        init_pipeline,
        save_attention_maps
    )
    ATTENTION_MAP_AVAILABLE = True
    print("✓ attention-map-diffusers successfully imported")
except ImportError as e:
    ATTENTION_MAP_AVAILABLE = False
    print(f"✗ attention-map-diffusers not available: {e}")
    print("Please install it with: pip install -e submodules/attention-map-diffusers")


@torch.no_grad()
def run_partcrafter_inference(
    pipe: Any,
    image_input: Union[str, Image.Image],
    num_parts: int,
    seed: int,
    num_tokens: int = 2048,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    max_num_expanded_coords: int = 1e9,
    use_flash_decoder: bool = False,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> tuple:
    """
    Run PartCrafter inference with optional attention visualization
    
    Returns:
        tuple: (outputs, processed_image)
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input)
    else:
        img_pil = image_input
    
    start_time = time.time()
    
    # Run inference
    outputs = pipe(
        image=[img_pil] * num_parts,
        attention_kwargs={"num_parts": num_parts},
        num_tokens=num_tokens,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_num_expanded_coords=max_num_expanded_coords,
        use_flash_decoder=use_flash_decoder,
    ).meshes
    
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    # Handle None outputs
    for i in range(len(outputs)):
        if outputs[i] is None:
            outputs[i] = trimesh.Trimesh(vertices=[[0, 0, 0]], faces=[[0, 0, 0]])
    
    return outputs, img_pil


def setup_attention_visualization(pipe: PartCrafterPipeline) -> bool:
    """
    Setup attention visualization for PartCrafter pipeline
    
    Args:
        pipe: PartCrafter pipeline
        
    Returns:
        bool: True if setup successful, False otherwise
    """
    if not ATTENTION_MAP_AVAILABLE:
        print("attention-map-diffusers not available, skipping attention visualization setup")
        return False
    
    try:
        # Initialize the pipeline for attention visualization
        # Note: PartCrafter uses a transformer-based architecture similar to SD3
        # We'll try to adapt the attention visualization for it
        
        print("Setting up attention visualization...")
        
        # Check if the pipeline has a transformer
        if hasattr(pipe, 'transformer'):
            print(f"Found transformer: {pipe.transformer.__class__.__name__}")
            
            # Try to initialize the pipeline for attention visualization
            # This might need adaptation for PartCrafter's specific architecture
            try:
                # For now, we'll manually set up the attention hooks
                # This is a simplified version - you might need to adapt based on PartCrafter's architecture
                
                # Register hooks for cross attention layers
                def hook_function(name, detach=True):
                    def forward_hook(module, input, output):
                        if hasattr(module.processor, "attn_map"):
                            timestep = getattr(module.processor, "timestep", 0)
                            attn_maps[timestep] = attn_maps.get(timestep, dict())
                            attn_maps[timestep][name] = module.processor.attn_map.cpu() if detach else module.processor.attn_map
                            del module.processor.attn_map
                    return forward_hook
                
                # Register hooks for cross attention layers
                for name, module in pipe.transformer.named_modules():
                    if 'attn' in name and hasattr(module, 'processor'):
                        if hasattr(module.processor, 'store_attn_map'):
                            module.processor.store_attn_map = True
                        hook = module.register_forward_hook(hook_function(name))
                
                print("✓ Attention visualization hooks registered")
                return True
                
            except Exception as e:
                print(f"✗ Failed to setup attention visualization: {e}")
                return False
        else:
            print("✗ No transformer found in pipeline")
            return False
            
    except Exception as e:
        print(f"✗ Error setting up attention visualization: {e}")
        return False


def save_attention_maps_for_partcrafter(attn_maps: dict, output_dir: str, case_name: str = "partcrafter"):
    """
    Save attention maps for PartCrafter (adapted version)
    
    Args:
        attn_maps: Attention maps dictionary
        output_dir: Output directory
        case_name: Case name for file naming
    """
    if not attn_maps:
        print("No attention maps to save")
        return
    
    attention_dir = os.path.join(output_dir, "attention_maps")
    os.makedirs(attention_dir, exist_ok=True)
    
    print(f"Saving attention maps to: {attention_dir}")
    
    # Save attention maps as numpy arrays and visualizations
    for timestep, layers in attn_maps.items():
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
                import matplotlib.pyplot as plt
                import seaborn as sns
                
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
                plt.title(f"Attention Heatmap - {layer_name}")
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                
                # Save heatmap
                heatmap_path = os.path.join(layer_dir, "attention_heatmap.png")
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"  Saved attention map for {layer_name}")
                
            except Exception as e:
                print(f"  Failed to create heatmap for {layer_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description='PartCrafter inference with attention visualization')
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--num_parts", type=int, required=True, help="Number of parts to generate")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--tag", type=str, default=None, help="Output tag")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_tokens", type=int, default=2048, help="Number of tokens")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale")
    parser.add_argument("--max_num_expanded_coords", type=int, default=1e9, help="Max expanded coordinates")
    parser.add_argument("--use_flash_decoder", action="store_true", help="Use flash decoder")
    parser.add_argument("--render", action="store_true", help="Generate renderings")
    parser.add_argument("--attention_viz", action="store_true", help="Enable attention visualization")
    parser.add_argument("--attention_viz_dir", type=str, default=None, help="Attention visualization directory")
    
    args = parser.parse_args()
    
    # Validate arguments
    MAX_NUM_PARTS = 8
    assert 1 <= args.num_parts <= MAX_NUM_PARTS, f"num_parts must be in [1, {MAX_NUM_PARTS}]"
    
    device = "cuda"
    dtype = torch.float16
    
    # Download pretrained weights
    print("Downloading PartCrafter weights...")
    partcrafter_weights_dir = "pretrained_weights/PartCrafter-Scene"
    snapshot_download(repo_id="wgsxm/PartCrafter-Scene", local_dir=partcrafter_weights_dir)
    
    # Initialize PartCrafter pipeline
    print("Initializing PartCrafter pipeline...")
    pipe: PartCrafterPipeline = PartCrafterPipeline.from_pretrained(partcrafter_weights_dir).to(device, dtype)
    
    # Setup attention visualization if requested
    attention_setup_success = False
    if args.attention_viz:
        attention_setup_success = setup_attention_visualization(pipe)
    
    set_seed(args.seed)
    
    # Run inference
    print(f"Running inference with {args.num_parts} parts...")
    outputs, processed_image = run_partcrafter_inference(
        pipe,
        image_input=args.image_path,
        num_parts=args.num_parts,
        seed=args.seed,
        num_tokens=args.num_tokens,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        max_num_expanded_coords=args.max_num_expanded_coords,
        use_flash_decoder=args.use_flash_decoder,
        dtype=dtype,
        device=device,
    )
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.tag is None:
        args.tag = time.strftime("%Y%m%d_%H_%M_%S")
    
    export_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(export_dir, exist_ok=True)
    
    # Save individual parts
    print("Saving generated parts...")
    for i, mesh in enumerate(outputs):
        mesh.export(os.path.join(export_dir, f"part_{i:02}.glb"))
    
    # Save merged mesh
    merged_mesh = get_colored_mesh_composition(outputs)
    merged_mesh.export(os.path.join(export_dir, "object.glb"))
    print(f"Generated {len(outputs)} parts and saved to {export_dir}")
    
    # Save attention maps if available
    if args.attention_viz and attention_setup_success and attn_maps:
        print("Saving attention maps...")
        attention_viz_dir = args.attention_viz_dir or os.path.join(export_dir, "attention_viz")
        save_attention_maps_for_partcrafter(attn_maps, attention_viz_dir, args.tag)
    
    # Generate renderings if requested
    if args.render:
        print("Generating renderings...")
        num_views = 36
        radius = 4
        fps = 18
        
        rendered_images = render_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_normals = render_normal_views_around_mesh(
            merged_mesh,
            num_views=num_views,
            radius=radius,
        )
        rendered_grids = make_grid_for_images_or_videos(
            [
                [processed_image] * num_views,
                rendered_images,
                rendered_normals,
            ], 
            nrow=3
        )
        
        # Save renderings
        export_renderings(
            rendered_images,
            os.path.join(export_dir, "rendering.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_normals,
            os.path.join(export_dir, "rendering_normal.gif"),
            fps=fps,
        )
        export_renderings(
            rendered_grids,
            os.path.join(export_dir, "rendering_grid.gif"),
            fps=fps,
        )
        
        # Save single frames
        rendered_image, rendered_normal, rendered_grid = rendered_images[0], rendered_normals[0], rendered_grids[0]
        rendered_image.save(os.path.join(export_dir, "rendering.png"))
        rendered_normal.save(os.path.join(export_dir, "rendering_normal.png"))
        rendered_grid.save(os.path.join(export_dir, "rendering_grid.png"))
        print("Renderings completed.")
    
    print(f"All outputs saved to: {export_dir}")


if __name__ == "__main__":
    main()
