#!/usr/bin/env python3
"""
PartCrafter DPO Training Script
Based on DSO's finetune.py but adapted for PartCrafter with penetration score preferences
"""

from copy import deepcopy
import datetime
from glob import glob
import inspect
import logging
import math
import os
from typing import Optional, Union, List, Dict, Any
import json

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
import argparse
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
import numpy as np
import sys
sys.path.append('/home/ran.ding/messy-kitchen/PartCrafter')

# PartCrafter imports
from src.models.transformers import PartCrafterDiTModel
from src.models.autoencoders import TripoSGVAEModel
from src.pipelines.pipeline_partcrafter import PartCrafterPipeline
from transformers import Dinov2Model, BitImageProcessor

# Use standard Python logger for dataset class
dataset_logger = logging.getLogger(__name__)
dataset_logger.setLevel(logging.INFO)

torch.autograd.set_detect_anomaly(True)

os.environ["WANDB_API_KEY"] = "ce1816bb45de1fc10f92c8bc17f2d7cc9b1a8757"


def create_logging(logging, logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)


def get_partcrafter_model(checkpoint_path: str):
    """Load PartCrafter model and trained checkpoints"""
    # Load PartCrafter DiT model from transformer subdirectory
    transformer = PartCrafterDiTModel.from_pretrained(checkpoint_path, subfolder="transformer")
    
    # Load VAE (TripoSG VAE) from vae subdirectory
    vae = TripoSGVAEModel.from_pretrained(checkpoint_path, subfolder="vae")
    
    # Load image encoder (DINOv2) from image_encoder_dinov2 subdirectory
    image_encoder = Dinov2Model.from_pretrained(checkpoint_path, subfolder="image_encoder_dinov2")
    image_processor = BitImageProcessor.from_pretrained(checkpoint_path, subfolder="feature_extractor_dinov2")
    
    return {
        "transformer": transformer,
        "vae": vae, 
        "image_encoder": image_encoder,
        "image_processor": image_processor
    }


# def create_output_folders(output_dir, config, exp_name):
#     """
#     Create an experiment output directory with timestamp,
#     including a subfolder for samples and a saved config.yaml.

#     Args:
#         output_dir (str): Base output directory
#         config (dict or OmegaConf): Configuration to save
#         exp_name (str): Experiment name

#     Returns:
#         str: Path to the created experiment directory
#     """
#     # Generate timestamp for unique directory name
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
#     out_dir = os.path.join(output_dir, f"{exp_name}_{timestamp}")

#     # Create main directory and 'samples' subfolder
#     os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

#     # Create a simple, safe config without complex OmegaConf operations
#     safe_config = {
#         "experiment_name": exp_name,
#         "timestamp": timestamp,
#         "output_dir": output_dir
#     }

#     # Save simple config to YAML file using yaml directly
#     import yaml
#     with open(os.path.join(out_dir, "config.yaml"), 'w') as f:
#         yaml.dump(safe_config, f, default_flow_style=False)

#     return out_dir

import os
import datetime
import yaml
from typing import Any

def _strip_unserializable(obj: Any, exclude_keys=({"accelerator", "_", "config"})):
    """
    Recursively:
      - drops keys that are known to be problematic (exclude_keys)
      - converts non-primitive objects to strings so YAML can dump them
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in exclude_keys:
                continue
            out[str(k)] = _strip_unserializable(v, exclude_keys)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_strip_unserializable(v, exclude_keys) for v in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        # Fallback for objects (e.g., Accelerate's Accelerator)
        return str(obj)

def create_output_folders(output_dir, config, exp_name):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out_dir = os.path.join(output_dir, f"{exp_name}_{now}")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)

    # Make a YAML-safe copy of config and save it
    safe_cfg = _strip_unserializable(config)
    cfg_path = os.path.join(out_dir, "config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(safe_cfg, f, sort_keys=False, allow_unicode=True)

    return out_dir


def sample_flow_matching_t_for_training(logit_normal_mu, logit_normal_sigma, bsz):
    """
    Sample t for flow matching training with a LogitNormal distribution.
    """
    t = torch.randn(bsz)
    t = torch.sigmoid(logit_normal_mu + t * logit_normal_sigma)
    return t


def forward_flow_matching_loss_v1(model, latent_features, t, cond_features, eps=None, num_parts=None, **kwargs):
    """Forward flow matching loss for PartCrafter latent features"""
    if eps is None:
        eps = torch.randn_like(latent_features)
    
    # Flow matching interpolation
    xt = (1 - t.view(t.shape + (1,) * (latent_features.ndim - 1))) * latent_features + \
         t.view(t.shape + (1,) * (latent_features.ndim - 1)) * eps
    
    target = eps - latent_features
    
    # Add num_parts to kwargs if provided
    if num_parts is not None:
        # num_parts should be a tensor with shape [batch_size] where each element is the number of parts for that sample
        if not isinstance(num_parts, torch.Tensor):
            num_parts = torch.tensor(num_parts, device=latent_features.device)
        # Ensure num_parts is on the correct device and has the right shape
        num_parts = num_parts.to(device=latent_features.device)
        # Convert to int tensor to ensure scalar values
        num_parts = num_parts.int()
        num_parts = num_parts.flatten()
        # Keep num_parts as tensor for PartCrafter transformer
        # PartCrafter transformer expects num_parts to be a tensor
        kwargs['attention_kwargs'] = {'num_parts': num_parts}
    
    pred = model(xt, t * 1000, cond_features, **kwargs)  ## TODO: Think of the cond_features' dimension, whether to concatenate or not?
    loss = (pred - target).pow(2).mean(dim=tuple(range(1, latent_features.ndim)))   ## TODO: think of the latent features' dimension
    return loss

def forward_flow_matching_loss(model, latent_win, latent_loss, t, cond_win, cond_loss, eps=None, num_parts=None, **kwargs):
    """Forward flow matching loss for PartCrafter with separate win/loss inputs"""
    
    def _clear_cuda():
        """Optimized CUDA memory cleanup"""
        import gc
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
    
    def compute_loss(latent_features, cond_features, t, eps, num_parts, **kwargs):
        """Helper function to compute flow matching loss for a single mode"""
        # Use autocast for memory efficiency
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            if eps is None:
                eps = torch.randn_like(latent_features)
            
            # Flow matching interpolation
            xt = (1 - t.view(t.shape + (1,) * (latent_features.ndim - 1))) * latent_features + \
                 t.view(t.shape + (1,) * (latent_features.ndim - 1)) * eps
            
            target = eps - latent_features
            
            # Add num_parts to kwargs if provided
            if num_parts is not None:
                # num_parts should be a tensor with shape [batch_size] where each element is the number of parts for that sample
                if not isinstance(num_parts, torch.Tensor):
                    num_parts = torch.tensor(num_parts, device=latent_features.device)
                # Ensure num_parts is on the correct device and has the right shape
                num_parts = num_parts.to(device=latent_features.device)
                # Convert to int tensor to ensure scalar values
                num_parts = num_parts.int()
                num_parts = num_parts.flatten()
                # Keep num_parts as tensor for PartCrafter transformer
                # PartCrafter transformer expects num_parts to be a tensor
                kwargs['attention_kwargs'] = {'num_parts': num_parts}
            
            pred = model(xt, t * 1000, cond_features, **kwargs)
            loss = (pred.sample - target).pow(2).mean(dim=tuple(range(1, latent_features.ndim)))
            
            # Clear intermediate tensors
            del xt, target
            if 'pred' in locals():
                del pred
        
        return loss
    
    # Mode 1: Compute loss for winning samples (latent_win, cond_win)
    _clear_cuda()
    loss_win = compute_loss(latent_win, cond_win, t, eps, num_parts, **kwargs)
    
    # Mode 2: Compute loss for losing samples (latent_loss, cond_loss)  
    _clear_cuda()
    loss_loser = compute_loss(latent_loss, cond_loss, t, eps, num_parts, **kwargs)
    _clear_cuda()
    
    return loss_win, loss_loser


def forward_dpo_loss(model, ref_model, latent_win, latent_loss, t, cond_features, beta, sample_same_epsilon, num_parts=None, **kwargs):
    """DPO loss for PartCrafter with penetration score preferences"""
    # Use the same conditioning for both win and loss (as they come from the same scene)
    cond_win = cond_loss = cond_features
    
    # 1. Forward pass with model
    loss_w, loss_l = forward_flow_matching_loss(
        model, latent_win, latent_loss, t, cond_win, cond_loss, 
        num_parts=num_parts, **kwargs
    )
    
    # 2. Forward pass with reference model
    with torch.no_grad():
        loss_w_ref, loss_l_ref = forward_flow_matching_loss(
            ref_model, latent_win, latent_loss, t, cond_win, cond_loss,
            eps=None if sample_same_epsilon else None, num_parts=num_parts, **kwargs
        )

    model_diff = loss_w - loss_l
    ref_diff = loss_w_ref - loss_l_ref

    inside_term = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term)
    return loss.mean()


class PartCrafterDPODataset(torch.utils.data.Dataset):
    """Dataset for PartCrafter DPO training with penetration score preferences"""
    
    def __init__(self, preference_pairs_file: str, max_samples: int = None, max_num_parts: int = 4):
        """
        Args:
            preference_pairs_file: Path to dpo_preference_pairs.json
            max_samples: Maximum number of samples to use (None for all)
            max_num_parts: Maximum number of parts allowed (filter out samples with more parts)
        """
        with open(preference_pairs_file, 'r') as f:
            data = json.load(f)
        
        self.preference_pairs = data['preference_pairs']
        
        # Filter out samples with num_parts > max_num_parts
        filtered_pairs = []
        for pair in self.preference_pairs:
            # Check num_parts from win_seed_dir
            win_seed_dir = pair['win_seed_dir']
            win_latent_paths = [f for f in os.listdir(win_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
            num_parts = len(win_latent_paths)
            
            if num_parts <= max_num_parts:
                filtered_pairs.append(pair)
            else:
                dataset_logger.debug(f"Filtered out sample with {num_parts} parts (max: {max_num_parts})")
        
        self.preference_pairs = filtered_pairs
        
        if max_samples:
            self.preference_pairs = self.preference_pairs[:max_samples]
        
        dataset_logger.info(f"Loaded {len(self.preference_pairs)} preference pairs for DPO training (filtered from {len(data['preference_pairs'])} total)")
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        
        # Load winning latent features
        win_seed_dir = pair['win_seed_dir']
        win_cond_path = os.path.join(win_seed_dir, 'cond.pt')
        win_latent_paths = [f for f in os.listdir(win_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
        
        # Load losing latent features  
        loss_seed_dir = pair['loss_seed_dir']
        loss_cond_path = os.path.join(loss_seed_dir, 'cond.pt')
        loss_latent_paths = [f for f in os.listdir(loss_seed_dir) if f.startswith('latent_sample_') and f.endswith('.pt')]
        
        # Load conditioning features (same for both) - use half precision to save memory
        cond_features = torch.load(win_cond_path, map_location='cpu', weights_only=True)
        cond_features = cond_features.half()  # Use half precision instead of float32

        # Load latent features with memory optimization
        win_latent_features = []
        for latent_path in sorted(win_latent_paths):
            latent = torch.load(os.path.join(win_seed_dir, latent_path), map_location='cpu', weights_only=True)
            win_latent_features.append(latent.half())  # Use half precision
        win_latent_features = torch.stack(win_latent_features, dim=0)
        
        loss_latent_features = []
        for latent_path in sorted(loss_latent_paths):
            latent = torch.load(os.path.join(loss_seed_dir, latent_path), map_location='cpu', weights_only=True)
            loss_latent_features.append(latent.half())  # Use half precision
        loss_latent_features = torch.stack(loss_latent_features, dim=0)
        
        # Infer num_parts from latent features shape
        num_parts = torch.tensor(win_latent_features.shape[0])
        n = int(num_parts.item()) 
        cond_features = cond_features.expand(n, -1, -1).contiguous() 

        return {
            'cond_features': cond_features,
            'latent_win': win_latent_features,  # Already half precision
            'latent_loss': loss_latent_features,  # Already half precision
            'penetration_diff': pair['penetration_diff'],
            'scene_name': pair['scene_name'],
            'win_seed': pair['win_seed'],
            'loss_seed': pair['loss_seed'],
            'num_parts': num_parts
        }


def main_training(
    # Dataset parameters
    preference_pairs_file: str = "/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json",
    max_samples: int = None,
    
    # Model parameters
    checkpoint_path: str = "/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512/checkpoints/017000",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    
    # Training parameters
    batch_size: int = 1,
    learning_rate: float = 5e-5,
    scale_lr: bool = False,
    lr_warmup_steps: int = 500,
    use_adafactor: bool = False,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 0.01,
    adam_epsilon: float = 1e-08,
    max_train_steps: int = 10000,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    
    # Memory optimization parameters
    use_gradient_checkpointing: bool = True,
    mixed_precision: str = "fp16",  # "fp16", "bf16", or "no"
    dataloader_num_workers: int = 2,  # Reduce from 4 to 2
    pin_memory: bool = True,
    prefetch_factor: int = 1,  # Reduce from 2 to 1
    
    # Flow matching parameters
    flow_matching_t_logit_normal_mu: float = 0.0,
    flow_matching_t_logit_normal_sigma: float = 1.0,
    
    # DPO parameters
    dpo_beta: float = 0.1,
    sample_same_epsilon: bool = True,
    
    # Logging and saving
    log_interval: int = 10,
    save_interval: int = 1000,
    ckpt_interval: int = 1000,
    
    # General parameters
    seed: int = 42,
    output_dir: str = "./outputs",
    exp_name: str = "partcrafter_dpo",
    logger_type: str = "tensorboard",
    resume_from_checkpoint: str = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=logger_type,
        project_dir=output_dir,
    )
    logger = get_logger(__name__, log_level="INFO")
    create_logging(logging, logger, accelerator)
    if seed is not None:
        set_seed(seed)

    if accelerator.is_main_process:
        run_dir = create_output_folders(output_dir, config, exp_name)

    if scale_lr:
        learning_rate = learning_rate * accelerator.num_processes * gradient_accumulation_steps * batch_size

    # Load PartCrafter model
    model_components = get_partcrafter_model(checkpoint_path)
    model = model_components["transformer"]
    
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "to_out.2"]  # PartCrafter-specific modules
        )
        model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory optimization
    if use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled for memory optimization")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, 
        betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay, eps=adam_epsilon
    ) if not use_adafactor else transformers.Adafactor(
        model.parameters(), lr=learning_rate, eps=adam_epsilon, weight_decay=adam_weight_decay,
        clip_threshold=1.0, scale_parameter=False, relative_step=False
    )

    lr_scheduler = get_scheduler(
        "constant_with_warmup", optimizer=optimizer, 
        num_warmup_steps=lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Load DPO dataset
    train_dataset = PartCrafterDPODataset(
        preference_pairs_file=preference_pairs_file,
        max_samples=max_samples,
        max_num_parts=5 # Filter out samples with more than 3 parts
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=dataloader_num_workers, 
        pin_memory=pin_memory, 
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Drop incomplete batches to avoid memory issues
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
    
    # Create reference model for DPO
    ref_model = deepcopy(model)
    ref_model.requires_grad_(False)

    if accelerator.is_main_process:
        # Initialize tracking
        clean_config = deepcopy(config)
        problematic_keys = [
            "accelerator", "__pydevd_ret_val_dict", "config", "_",
            "model", "optimizer", "lr_scheduler", "train_loader", "ref_model"
        ]
        for k in problematic_keys:
            clean_config.pop(k, None)
        
        whitelist = {
            'exp_name', 'output_dir', 'preference_pairs_file', 'max_samples', 'checkpoint_path',
            'batch_size', 'learning_rate', 'scale_lr', 'lr_warmup_steps', 'use_adafactor',
            'adam_beta1', 'adam_beta2', 'adam_weight_decay', 'adam_epsilon',
            'max_train_steps', 'max_grad_norm', 'gradient_accumulation_steps',
            'flow_matching_t_logit_normal_mu', 'flow_matching_t_logit_normal_sigma',
            'dpo_beta', 'sample_same_epsilon', 'log_interval', 'save_interval', 'ckpt_interval',
            'seed', 'logger_type', 'resume_from_checkpoint',
            'use_lora', 'lora_r', 'lora_alpha', 'lora_dropout'
        }
        wandb_config = {k: v for k, v in clean_config.items() if k in whitelist}
        
        accelerator.init_trackers(project_name="partcrafter_dpo", config=wandb_config, init_kwargs={"wandb": {"name": exp_name}})

    total_batch_size = accelerator.num_processes * gradient_accumulation_steps * batch_size
    num_train_epochs = math.ceil(max_train_steps * gradient_accumulation_steps / len(train_loader))

    if resume_from_checkpoint is not None:
        global_step = int(resume_from_checkpoint.split("-")[-1])
        accelerator.load_state(resume_from_checkpoint, strict=False)
    else:
        global_step = 0

    logger.info(f"Model loaded! {model}, num params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Total batch size: {total_batch_size}")
    logger.info(f"Number of training epochs: {num_train_epochs}")
    logger.info(f"Number of preference pairs: {len(train_dataset)}")

    # Training loop with memory optimization
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    def clear_memory():
        """Clear GPU memory"""
        import gc
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()

    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                # Clear memory at the start of each step
                if step % 10 == 0:  # Clear memory every 10 steps
                    clear_memory()
                
                num_parts = batch['num_parts']
                num_parts = num_parts.squeeze(0)
                num_parts_int = num_parts.int()
                
                # Sample t for flow matching
                t = sample_flow_matching_t_for_training(
                    flow_matching_t_logit_normal_mu,
                    flow_matching_t_logit_normal_sigma,
                    num_parts_int
                ).to(accelerator.device)

                # Extract batch data
                cond_features = batch['cond_features']
                latent_win = batch['latent_win']
                latent_loss = batch['latent_loss']
                penetration_diff = batch['penetration_diff']
                
                # Ensure num_parts is properly formatted for the batch
                if isinstance(num_parts, list):
                    num_parts = torch.stack(num_parts)
                elif num_parts.dim() == 0:
                    num_parts = num_parts.unsqueeze(0)
                
                latent_win = latent_win.squeeze(0)
                latent_loss = latent_loss.squeeze(0)
                cond_features = cond_features.squeeze(0)

                # Compute DPO loss with memory optimization
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    loss = forward_dpo_loss(
                        model=model,
                        ref_model=ref_model,
                        latent_win=latent_win,
                        latent_loss=latent_loss,
                        t=t,
                        cond_features=cond_features,
                        beta=dpo_beta,
                        sample_same_epsilon=sample_same_epsilon,
                        num_parts=num_parts
                    )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Clear intermediate variables
                del latent_win, latent_loss, cond_features, t, num_parts
                if 'loss' in locals():
                    del loss

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % log_interval == 0:
                    accelerator.log({
                        "train/loss": loss.item() if 'loss' in locals() else 0.0,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/penetration_diff": penetration_diff.mean().item(),
                        "train/global_step": global_step,
                    }, step=global_step)

                if global_step % save_interval == 0:
                    if accelerator.is_main_process:
                        # Save checkpoint
                        save_path = os.path.join(run_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")
                        clear_memory()  # Clear memory after saving

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.end_training()

    if accelerator.is_main_process:
        # Save final model
        final_save_path = os.path.join(run_dir, "final_model")
        accelerator.save_state(final_save_path)
        logger.info(f"Training completed! Final model saved to {final_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PartCrafter DPO Training")
    
    # Dataset parameters
    parser.add_argument("--preference_pairs_file", type=str, 
                       default="/home/ran.ding/messy-kitchen/dso/messy_kitchen_data/dpo_data/messy_kitchen_configs/dpo_preference_pairs.json",
                       help="Path to DPO preference pairs JSON file")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to use (None for all)")
    
    # Model parameters
    parser.add_argument("--checkpoint_path", type=str,
                       default="/home/ran.ding/messy-kitchen/dso/submodules/partcrafter_ran/runs/messy_kitchen/part_1/messy_kitchen_part1_mp8_nt512/checkpoints/017000",
                       help="Path to PartCrafter checkpoint")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_train_steps", type=int, default=10000,
                       help="Maximum training steps")
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                       help="DPO beta parameter")
    
    # Memory optimization parameters
    parser.add_argument("--use_gradient_checkpointing", action="store_true", default=True,
                       help="Use gradient checkpointing for memory optimization")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["fp16", "bf16", "no"],
                       help="Mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=2,
                       help="Number of dataloader workers")
    parser.add_argument("--prefetch_factor", type=int, default=1,
                       help="Dataloader prefetch factor")
    
    # General parameters
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--exp_name", type=str, default="partcrafter_dpo",
                       help="Experiment name")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    main_training(**vars(args))


