# PartCrafter DPO Implementation Plan

## Overview
This document outlines the implementation plan for adapting DSO's DPO (Direct Preference Optimization) finetuning approach to PartCrafter. The goal is to improve PartCrafter's 3D mesh generation quality by training on preference data, using **penetration score** as the evaluation metric instead of DSO's stability metrics.

### Data Sources
- **Input Images**: `messy_kitchen_data/messy_kitchen_scenes_renderings/` - Contains rendering images as model inputs
- **Ground Truth Meshes**: `messy_kitchen_data/raw_messy_kitchen_scenes/` - Contains GLB mesh files for evaluation
- **Penetration Score**: Reference implementation in `submodules/TAGAC/messy_kitchen_scripts/calculate_penetration_score.py`

## Project Structure Analysis

### DSO Components (Reference)
- **Training Script**: `finetune.py` - Main DPO training implementation
- **Data Processing**: `data_preprocessing/generate_synthetic_data.py` - Generate 3D models from images
- **Simulation Feedback**: `data_preprocessing/augment_with_simulation_feedback.py` - Add physical stability scores
- **Dataset**: `dataset.py` - Custom dataset for DPO training
- **Configuration**: `configs/dpo.yaml` - Training hyperparameters

### PartCrafter Components (Target)
- **Training Script**: `src/train_partcrafter.py` - Current training implementation
- **Data Creation**: `scripts/create_dpo_data.py` - Existing DPO data creation script
- **Pipeline**: `src/pipelines/pipeline_partcrafter.py` - Inference pipeline
- **Models**: `src/models/transformers/partcrafter_transformer.py` - Main transformer model
- **Configuration**: `configs/mp8_nt512.yaml` - Current training config

## Implementation Plan

### Phase 1: Dataset Preparation Flow

#### 1.1 DPO Dataset Creation
**File**: `scripts/create_partcrafter_dpo_data.py` (new)

**Purpose**: Generate diverse 3D mesh outputs using PartCrafter with different random seeds from messy_kitchen_data, then evaluate them for penetration scores using TAGAC's well-defined penetration level calculation.

**Data Flow**:
1. **Input**: `messy_kitchen_data/messy_kitchen_scenes_renderings/` (rendering images)
2. **Generation**: PartCrafter with different seeds → multiple mesh candidates
3. **Evaluation**: TAGAC penetration score calculation
4. **Output**: DPO preference pairs based on penetration scores

**Key Components**:
```python
class PartCrafterDPODataCreator:
    def __init__(self, model_path: str, device: str = "cuda"):
        # Load PartCrafter pipeline and trained checkpoints
        self.pipeline = PartCrafterPipeline.from_pretrained(model_path)
        self.pipeline.cuda()
        
        # Initialize penetration evaluator using TAGAC implementation
        self.penetration_evaluator = PenetrationScoreEvaluator()
        
    def generate_candidates(self, image_path: str, num_candidates: int = 4):
        # Generate multiple 3D meshes with different seeds
        candidates = []
        for seed in range(num_candidates):
            mesh = self.pipeline(image_path, seed=seed)
            candidates.append(mesh)
        return candidates
    
    def evaluate_penetration(self, mesh_path: str, gt_mesh_path: str = None) -> Dict[str, float]:
        # Use TAGAC's well-defined penetration level calculation
        if gt_mesh_path:
            # Compare with ground truth from raw_messy_kitchen_scenes
            return self.penetration_evaluator.compute_penetration_level_detailed(
                mesh_list=[mesh_path], 
                merged_mesh=mesh_path, 
                size=0.3, 
                resolution=40
            )
        else:
            # Self-penetration analysis
            return self.penetration_evaluator.compute_penetration_level(mesh_path)
    
    def create_preference_pairs(self, candidates: List, scores: List[Dict[str, float]]):
        # Create win/loss pairs based on penetration_level scores
        # Lower penetration_level = better (less penetration)
        penetration_levels = [score['penetration_level'] for score in scores]
        return self._create_pairs_from_penetration_scores(candidates, penetration_levels)
```

**Directory Structure**:
```
dpo_data/
├── messy_kitchen_dpo/
│   ├── 00a9f565dea8439d8f5fbc403e490bff_combined/
│   │   ├── 000/  # seed 0
│   │   │   ├── pred_merged.glb
│   │   │   ├── pred_obj_*.ply  # individual meshes
│   │   │   ├── penetration_analysis.json
│   │   │   ├── seed_info.json
│   │   │   └── latent_features.pt
│   │   ├── 001/  # seed 1
│   │   ├── 002/  # seed 2
│   │   ├── 003/  # seed 3
│   │   ├── input_image.jpg  # from rendering
│   │   ├── gt_mesh.glb      # from raw_messy_kitchen_scenes
│   │   ├── preference_pairs.json
│   │   └── scene_metadata.json
│   └── 00e4693b4e934b849e1b9f79e7e8682e_combined/
```

**Data Mapping**:
- **Input Images**: `messy_kitchen_scenes_renderings/{scene_id}_combined/rendering.png`
- **Ground Truth**: `raw_messy_kitchen_scenes/{scene_id}_combined.glb`
- **Scene Info**: `messy_kitchen_scenes_renderings/{scene_id}_combined/num_parts.json`, `iou.json`

#### 1.2 Data Processing Pipeline
**File**: `data_preprocessing/partcrafter_dpo_dataset.py` (new)

**Purpose**: Process DPO data into training format, adapted from DSO's dataset.py for PartCrafter architecture

**Key Components**:

##### Data Structure Analysis (DSO vs PartCrafter)

**DSO Data Structure**:
```
train_data/train_outputs/train_demo/
├── category/
│   ├── prompt/
│   │   ├── 000_cond.pt                    # Image conditioning features
│   │   ├── 000_sparse_sample_000.pt       # Winning sparse structure (good stability)
│   │   ├── 000_sparse_sample_001.pt       # Losing sparse structure (bad stability)
│   │   ├── 000_angles.npy                 # Stability angles [N, 2]
│   │   ├── model_info.npz                 # Global model info
│   │   └── available_images.npy           # Valid image IDs
│   └── image.png
```

**PartCrafter DPO Data Structure**:
```
dpo_data/partcrafter_dataset/
├── category/
│   ├── prompt/
│   │   ├── 000_cond.pt                    # Image conditioning features (from PartCrafter VAE)
│   │   ├── 000_latent_win_000.pt          # Winning latent features (low penetration)
│   │   ├── 000_latent_loss_001.pt         # Losing latent features (high penetration)
│   │   ├── 000_mesh_win_000.glb           # Winning mesh file
│   │   ├── 000_mesh_loss_001.glb          # Losing mesh file
│   │   ├── 000_penetration_scores.npy     # Penetration scores [N, 1]
│   │   ├── 000_part_surfaces_win.pt       # Winning part surfaces [N_parts, P, 6]
│   │   ├── 000_part_surfaces_loss.pt      # Losing part surfaces [N_parts, P, 6]
│   │   ├── model_info.npz                 # Global model info
│   │   └── available_images.npy           # Valid image IDs
│   └── image.png
```

##### Core Dataset Class

```python
class PartCrafterDPODataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_dir: str = "./dpo_data",
        category: str = "partcrafter-dataset",
        prompts: Optional[list[str]] = None,
        num_images_per_prompt: int = 10,
        image_ids: Optional[list[int]] = None,
        num_models_per_image: int = 4,
        penetration_threshold: float = 0.1,
        sample_from_all_multiviews: bool = False,
    ):
        self.dataset_root = osp.join(root_dir, category)
        self.prompts = prompts if prompts is not None else [
            d for d in os.listdir(self.dataset_root) 
            if os.path.exists(osp.join(self.dataset_root, d, "available_images.npy"))
        ]
        self.num_models_per_image = num_models_per_image
        self.penetration_threshold = penetration_threshold
        self.sample_from_all_multiviews = sample_from_all_multiviews
        
        # PartCrafter-specific settings
        self.image_size = (512, 512)
        self.max_num_parts = 8  # PartCrafter limit
        
    def __getitem__(self, index):
        if self.sample_from_all_multiviews:
            # Sample from all available multiviews (similar to DSO)
            return self._get_multiview_sample()
        else:
            # Standard sampling (similar to DSO)
            return self._get_standard_sample()
    
    def _get_multiview_sample(self):
        """Sample from all available images with preference pairs"""
        while True:
            try:
                prompt = random.choice(self.prompts)
                prompt_root = osp.join(self.dataset_root, prompt)
                
                if not osp.exists(osp.join(prompt_root, "model_info.npz")):
                    continue
                    
                model_info = np.load(osp.join(prompt_root, "model_info.npz"))
                image_ids = model_info["image_ids"]
                model_ids = model_info["model_ids"] 
                penetration_scores = model_info["penetration_scores"]
                
                # Find preference pairs (win/loss based on penetration scores)
                good_indices = indices[penetration_scores < self.penetration_threshold]
                bad_indices = indices[penetration_scores >= self.penetration_threshold]
                
                if len(good_indices) == 0 or len(bad_indices) == 0:
                    continue
                    
                # Sample winning and losing candidates
                good_index = np.random.choice(good_indices)
                bad_index = np.random.choice(bad_indices)
                
                image_id_win = image_ids[good_index]
                image_id_loss = image_ids[bad_index]
                model_win = model_ids[good_index]
                model_loss = model_ids[bad_index]
                
                # Load conditioning features
                cond_path = osp.join(prompt_root, f"{image_id_win:03d}_cond.pt")
                cond = torch.load(cond_path, weights_only=True, map_location="cpu")
                
                # Load winning and losing latent features
                latent_win_path = osp.join(prompt_root, f"{image_id_win:03d}_latent_win_{model_win:03d}.pt")
                latent_loss_path = osp.join(prompt_root, f"{image_id_loss:03d}_latent_loss_{model_loss:03d}.pt")
                
                latent_win = torch.load(latent_win_path, weights_only=True, map_location="cpu")
                latent_loss = torch.load(latent_loss_path, weights_only=True, map_location="cpu")
                
                # Load part surfaces for rendering
                part_surfaces_win_path = osp.join(prompt_root, f"{image_id_win:03d}_part_surfaces_win_{model_win:03d}.pt")
                part_surfaces_loss_path = osp.join(prompt_root, f"{image_id_loss:03d}_part_surfaces_loss_{model_loss:03d}.pt")
                
                part_surfaces_win = torch.load(part_surfaces_win_path, weights_only=True, map_location="cpu")
                part_surfaces_loss = torch.load(part_surfaces_loss_path, weights_only=True, map_location="cpu")
                
                return {
                    "prompt": prompt,
                    "cond": cond,                           # Image conditioning [H, W, 3]
                    "latent_win": latent_win,               # Winning latent features [N, D]
                    "latent_loss": latent_loss,             # Losing latent features [N, D]
                    "part_surfaces_win": part_surfaces_win, # Winning part surfaces [N_parts, P, 6]
                    "part_surfaces_loss": part_surfaces_loss, # Losing part surfaces [N_parts, P, 6]
                    "penetration_win": penetration_scores[good_index],
                    "penetration_loss": penetration_scores[bad_index],
                }
                
            except FileNotFoundError as e:
                pass
            except Exception as e:
                print(f"Error in multiview sampling: {e}")
    
    def _get_standard_sample(self):
        """Standard sampling from available images"""
        while True:
            prompt = random.choice(self.prompts)
            prompt_root = osp.join(self.dataset_root, prompt)
            
            if not osp.exists(osp.join(prompt_root, "available_images.npy")):
                continue
                
            available_images = np.load(osp.join(prompt_root, "available_images.npy"))
            image_id = random.choice(available_images)
            break
        
        # Load conditioning features
        cond_path = osp.join(prompt_root, f"{image_id:03d}_cond.pt")
        cond = torch.load(cond_path, weights_only=True, map_location="cpu")
        
        # Load penetration scores
        penetration_path = osp.join(prompt_root, f"{image_id:03d}_penetration_scores.npy")
        penetration_scores = np.load(penetration_path)
        
        # Find preference pairs
        good_model_ids = np.arange(self.num_models_per_image)
        winning_candidate_ids = good_model_ids[penetration_scores < self.penetration_threshold]
        loss_candidate_ids = good_model_ids[penetration_scores >= self.penetration_threshold]
        
        if len(winning_candidate_ids) == 0 or len(loss_candidate_ids) == 0:
            # Fallback: create preference pair artificially
            return self._create_artificial_preference_pair(prompt_root, image_id, cond, penetration_scores)
        
        model_win = np.random.choice(winning_candidate_ids)
        model_loss = np.random.choice(loss_candidate_ids)
        
        # Load latent features
        latent_win_path = osp.join(prompt_root, f"{image_id:03d}_latent_win_{model_win:03d}.pt")
        latent_loss_path = osp.join(prompt_root, f"{image_id:03d}_latent_loss_{model_loss:03d}.pt")
        
        latent_win = torch.load(latent_win_path, weights_only=True, map_location="cpu")
        latent_loss = torch.load(latent_loss_path, weights_only=True, map_location="cpu")
        
        # Load part surfaces
        part_surfaces_win_path = osp.join(prompt_root, f"{image_id:03d}_part_surfaces_win_{model_win:03d}.pt")
        part_surfaces_loss_path = osp.join(prompt_root, f"{image_id:03d}_part_surfaces_loss_{model_loss:03d}.pt")
        
        part_surfaces_win = torch.load(part_surfaces_win_path, weights_only=True, map_location="cpu")
        part_surfaces_loss = torch.load(part_surfaces_loss_path, weights_only=True, map_location="cpu")
        
        return {
            "prompt": prompt,
            "cond": cond,
            "latent_win": latent_win,
            "latent_loss": latent_loss,
            "part_surfaces_win": part_surfaces_win,
            "part_surfaces_loss": part_surfaces_loss,
            "penetration_win": penetration_scores[model_win],
            "penetration_loss": penetration_scores[model_loss],
        }
```

##### Key Differences from DSO Dataset

1. **Conditioning Format**: 
   - DSO: `cond` (image features) + `sparse_x0` (sparse structure)
   - PartCrafter: `cond` (image features) + `latent_win/loss` (VAE latent features)

2. **Preference Criteria**:
   - DSO: Stability angles (lower = better)
   - PartCrafter: Penetration scores (lower = better)

3. **Data Format**:
   - DSO: Sparse structure tensors
   - PartCrafter: Part surface tensors + latent features

4. **Rendering Support**:
   - PartCrafter includes `part_surfaces` for object-centric rendering
   - Supports colorful mesh visualization

##### Integration with PartCrafter Pipeline

```python
def load_partcrafter_conditioning(image_path: str, pipeline: PartCrafterPipeline):
    """Load and process image conditioning using PartCrafter pipeline"""
    image = Image.open(image_path).resize((512, 512))
    image = pipeline.preprocess_image(image)
    
    # Get VAE encoding (similar to DSO's cond processing)
    with torch.no_grad():
        image_cond = pipeline.vae.encode(image.unsqueeze(0)).latent_dist.sample()
        image_cond = image_cond.squeeze(0)  # Remove batch dimension
    
    return image_cond

def extract_partcrafter_latent_features(mesh_path: str, pipeline: PartCrafterPipeline):
    """Extract latent features from generated mesh using PartCrafter VAE"""
    # Load mesh and convert to PartCrafter format
    mesh = trimesh.load(mesh_path)
    part_surfaces = mesh_to_part_surfaces(mesh)  # Convert to [N_parts, P, 6]
    
    # Encode using PartCrafter VAE
    with torch.no_grad():
        latent_features = pipeline.vae.encode(part_surfaces).latent_dist.sample()
    
    return latent_features, part_surfaces
```

##### PartCrafter Latent Feature Extraction Details

**PartCrafter VAE Architecture Analysis**:

1. **Input Format**: PartCrafter expects point clouds with features `[N_parts, P, 6]` where:
   - `N_parts`: Number of parts/components
   - `P`: Number of points per part
   - `6`: [x, y, z, nx, ny, nz] (position + normal)

2. **Encoding Process**:
   ```python
   def encode_partcrafter_mesh(mesh_path: str, pipeline: PartCrafterPipeline, num_tokens: int = 2048):
       """Extract latent features from mesh using PartCrafter VAE"""
       # Load mesh
       mesh = trimesh.load(mesh_path)
       
       # Convert mesh to PartCrafter format
       # Method 1: Direct point sampling
       vertices = mesh.vertices  # [N, 3]
       normals = mesh.vertex_normals  # [N, 3]
       
       # Sample points if mesh is too dense
       if len(vertices) > num_tokens:
           indices = np.random.choice(len(vertices), num_tokens, replace=False)
           vertices = vertices[indices]
           normals = normals[indices]
       
       # Create part surfaces format [1, P, 6]
       part_surfaces = torch.cat([
           torch.from_numpy(vertices).float(),
           torch.from_numpy(normals).float()
       ], dim=-1).unsqueeze(0)  # Add batch dimension
       
       # Encode using PartCrafter VAE
       with torch.no_grad():
           # PartCrafter VAE expects [B, N, C] where C=6
           encoded = pipeline.vae.encode(part_surfaces, num_tokens=num_tokens)
           latent_features = encoded.latent_dist.sample()  # [B, num_tokens, latent_dim]
       
       return latent_features.squeeze(0), part_surfaces.squeeze(0)  # Remove batch dim
   ```

3. **Key Differences from DSO**:
   - **DSO**: Uses sparse structure tensors directly
   - **PartCrafter**: Uses VAE-encoded latent features from point clouds
   - **PartCrafter**: Supports hierarchical geometry extraction
   - **PartCrafter**: Uses TripoSG VAE with cross-attention encoding

##### Mesh to PartCrafter Format Conversion

```python
def mesh_to_partcrafter_format(mesh_path: str, num_tokens: int = 2048):
    """Convert mesh to PartCrafter-compatible format"""
    import trimesh
    import torch
    import numpy as np
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    # Extract vertices and normals
    vertices = mesh.vertices.astype(np.float32)
    normals = mesh.vertex_normals.astype(np.float32)
    
    # Ensure we have normals (compute if missing)
    if len(normals) == 0:
        mesh.compute_vertex_normals()
        normals = mesh.vertex_normals.astype(np.float32)
    
    # Sample points if needed
    if len(vertices) > num_tokens:
        indices = np.random.choice(len(vertices), num_tokens, replace=False)
        vertices = vertices[indices]
        normals = normals[indices]
    elif len(vertices) < num_tokens:
        # Repeat points if too few
        repeat_factor = num_tokens // len(vertices) + 1
        vertices = np.tile(vertices, (repeat_factor, 1))[:num_tokens]
        normals = np.tile(normals, (repeat_factor, 1))[:num_tokens]
    
    # Create part surfaces format [P, 6]
    part_surfaces = torch.cat([
        torch.from_numpy(vertices),
        torch.from_numpy(normals)
    ], dim=-1)
    
    return part_surfaces  # Shape: [num_tokens, 6]
```

### Phase 2: DPO Finetuning Training Script

#### 2.1 Main Training Script
**File**: `src/train_partcrafter_dpo.py` (new)

**Based on**: DSO's `finetune.py` + PartCrafter's `train_partcrafter.py`

#### 2.2 Detailed Modifications from DSO to PartCrafter

##### 2.2.1 Model Loading and Initialization

**DSO Original**:
```python
def get_model():
    sparse_structure_flow_model = models.from_pretrained(
        "JeffreyXiang/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16",
    )
    return sparse_structure_flow_model
```

**PartCrafter Adaptation**:
```python
def get_partcrafter_model(checkpoint_path: str):
    """Load PartCrafter model and trained checkpoints"""
    # Load PartCrafter DiT model
    transformer = PartCrafterDiTModel.from_pretrained(checkpoint_path)
    
    # Load VAE (TripoSG VAE)
    vae = TripoSGVAEModel.from_pretrained(checkpoint_path)
    
    # Load image encoder (DINOv2)
    image_encoder = Dinov2Model.from_pretrained("facebook/dinov2-large")
    image_processor = BitImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    return {
        "transformer": transformer,
        "vae": vae, 
        "image_encoder": image_encoder,
        "image_processor": image_processor
    }
```

##### 2.2.2 Pipeline and Data Loading

**DSO Original**:
```python
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
```

**PartCrafter Adaptation**:
```python
pipeline = PartCrafterPipeline.from_pretrained(checkpoint_path)
```

##### 2.2.3 Dataset Loading

**DSO Original**:
```python
from dataset import SyntheticDataset

train_dataset = SyntheticDataset(
    root_dir=dataset_dir,
    category=category,
    num_models_per_image=num_models_per_image,
    stable_threshold=stable_threshold,
    sample_from_all_multiviews=sample_from_all_multiviews,
)
```

**PartCrafter Adaptation**:
```python
from data_preprocessing.partcrafter_dpo_dataset import PartCrafterDPODataset

train_dataset = PartCrafterDPODataset(
    root_dir=dataset_dir,
    category=category,
    num_models_per_image=num_models_per_image,
    penetration_threshold=penetration_threshold,  # Changed from stable_threshold
    sample_from_all_multiviews=sample_from_all_multiviews,
)
```

##### 2.2.4 DPO Loss Function Adaptation

**DSO Original**:
```python
def forward_dpo_loss(model, ref_model, x0_win, x0_loss, t, cond, beta, sample_same_epsilon, **kwargs):
    # 0. Concatenate x0_win and x0_loss
    x0 = torch.cat([x0_win, x0_loss], dim=0)
    t = torch.cat([t, t], dim=0)
    cond = torch.cat([cond, cond], dim=0)

    # 1. Forward pass
    eps = torch.randn_like(x0)
    loss_w, loss_l = forward_flow_matching_loss(model, x0, t, cond, eps, **kwargs).chunk(2)
    with torch.no_grad():
        loss_w_ref, loss_l_ref = forward_flow_matching_loss(ref_model, x0, t, cond, eps=eps if sample_same_epsilon else None, **kwargs).detach().chunk(2)

    model_diff = loss_w - loss_l
    ref_diff = loss_w_ref - loss_l_ref

    inside_term = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term)
    return loss.mean()
```

**PartCrafter Adaptation**:
```python
def forward_dpo_loss_partcrafter(model, ref_model, latent_win, latent_loss, t, cond, beta, sample_same_epsilon, **kwargs):
    """
    DPO loss adapted for PartCrafter transformer architecture
    Args:
        model: PartCrafter DiT model (finetuned)
        ref_model: PartCrafter DiT model (reference/frozen)
        latent_win: Winning latent features [B, num_tokens, latent_dim]
        latent_loss: Losing latent features [B, num_tokens, latent_dim]
        t: Timesteps
        cond: Image conditioning features
        beta: DPO temperature parameter
    """
    # 0. Concatenate winning and losing latents
    latents = torch.cat([latent_win, latent_loss], dim=0)
    t = torch.cat([t, t], dim=0)
    cond = torch.cat([cond, cond], dim=0)

    # 1. Forward pass through PartCrafter transformer
    eps = torch.randn_like(latents)
    
    # PartCrafter uses different noise prediction format
    noise_pred_win, noise_pred_loss = model(
        latents, t, encoder_hidden_states=cond, **kwargs
    ).chunk(2)
    
    with torch.no_grad():
        ref_noise_pred_win, ref_noise_pred_loss = ref_model(
            latents, t, encoder_hidden_states=cond, **kwargs
        ).chunk(2)

    # 2. Compute flow matching losses
    target_win = eps[:len(eps)//2] - latent_win
    target_loss = eps[len(eps)//2:] - latent_loss
    
    loss_w = (noise_pred_win - target_win).pow(2).mean()
    loss_l = (noise_pred_loss - target_loss).pow(2).mean()
    
    with torch.no_grad():
        loss_w_ref = (ref_noise_pred_win - target_win).pow(2).mean()
        loss_l_ref = (ref_noise_pred_loss - target_loss).pow(2).mean()

    # 3. Compute DPO loss
    model_diff = loss_w - loss_l
    ref_diff = loss_w_ref - loss_l_ref

    inside_term = -0.5 * beta * (model_diff - ref_diff)
    loss = -F.logsigmoid(inside_term)
    return loss.mean()
```

##### 2.2.5 Training Loop Modifications

**DSO Original**:
```python
for step, batch in enumerate(train_loader):
    with accelerator.accumulate(model):
        with accelerator.autocast():
            # Ensure data types are consistent
            batch['model_win_sparse_x0'] = batch['model_win_sparse_x0'].to(dtype=torch.float32)
            batch['model_loss_sparse_x0'] = batch['model_loss_sparse_x0'].to(dtype=torch.float32)
            batch['cond'] = batch['cond'].to(dtype=torch.float32)
            t = t.to(dtype=torch.float32)
            
            if use_dpo:
                loss = forward_dpo_loss(
                    model=model,
                    ref_model=ref_model,
                    x0_win=batch['model_win_sparse_x0'],
                    x0_loss=batch['model_loss_sparse_x0'],
                    t=t,
                    cond=batch['cond'],
                    beta=dpo_beta,
                    sample_same_epsilon=sample_same_epsilon,
                )
```

**PartCrafter Adaptation**:
```python
for step, batch in enumerate(train_loader):
    with accelerator.accumulate(model):
        with accelerator.autocast():
            # Ensure data types are consistent
            batch['latent_win'] = batch['latent_win'].to(dtype=torch.float32)
            batch['latent_loss'] = batch['latent_loss'].to(dtype=torch.float32)
            batch['cond'] = batch['cond'].to(dtype=torch.float32)
            t = t.to(dtype=torch.float32)
            
            if use_dpo:
                loss = forward_dpo_loss_partcrafter(
                    model=model,
                    ref_model=ref_model,
                    latent_win=batch['latent_win'],
                    latent_loss=batch['latent_loss'],
                    t=t,
                    cond=batch['cond'],
                    beta=dpo_beta,
                    sample_same_epsilon=sample_same_epsilon,
                )
```

##### 2.2.6 LoRA Configuration

**DSO Original**:
```python
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["to_q", "to_kv", "to_out", "to_qkv"]
)
```

**PartCrafter Adaptation**:
```python
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    # PartCrafter DiT uses different attention module names
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "to_out.2"]  # Check actual module names
)
```

##### 2.2.7 Configuration File Changes

**Key Configuration Differences**:

1. **Model Paths**:
   - DSO: `"JeffreyXiang/TRELLIS-image-large"`
   - PartCrafter: `"path/to/trained/partcrafter/checkpoint"`

2. **Dataset Settings**:
   - DSO: `stable_threshold: 20.0`
   - PartCrafter: `penetration_threshold: 0.1`

3. **Model Architecture**:
   - DSO: `sparse_structure_flow_model`
   - PartCrafter: `transformer`, `vae`, `image_encoder`

4. **Data Format**:
   - DSO: `sparse_x0` tensors
   - PartCrafter: `latent_win/loss` tensors

##### 2.2.8 Import Statements Changes

**DSO Original**:
```python
from trellis import models
from trellis.pipelines import TrellisImageTo3DPipeline
from dataset import SyntheticDataset
```

**PartCrafter Adaptation**:
```python
from src.models.transformers import PartCrafterDiTModel
from src.models.autoencoders import TripoSGVAEModel
from src.pipelines import PartCrafterPipeline
from transformers import Dinov2Model, BitImageProcessor
from data_preprocessing.partcrafter_dpo_dataset import PartCrafterDPODataset
```

#### 2.2 Training Configuration
**File**: `configs/partcrafter_dpo.yaml` (new)

**Based on**: DSO's `configs/dpo.yaml` + PartCrafter's `configs/mp8_nt512.yaml`

```yaml
# PartCrafter-specific settings
model:
  pretrained_model_name_or_path: 'pretrained_weights/PartCrafter'
  checkpoint_path: 'path/to/trained/partcrafter/checkpoint'
  vae:
    num_tokens: 512
  transformer:
    enable_local_cross_attn: true
    global_attn_block_ids: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# DPO settings (adapted from DSO)
dpo:
  beta: 500.0
  use_dpo: true
  sample_same_epsilon: true

# LoRA settings
lora:
  use_lora: true
  lora_r: 64
  lora_alpha: 128
  lora_dropout: 0.0

# Training settings
train:
  batch_size_per_gpu: 2
  learning_rate: 5e-6
  max_train_steps: 8000
  gradient_accumulation_steps: 2
  log_interval: 10
  ckpt_interval: 2000

# Dataset settings
dataset:
  dpo_data_dir: "dpo_data/messy_kitchen_dpo"
  num_models_per_image: 4
  penetration_threshold: 0.1  # Threshold for win/loss determination
  input_images_dir: "../messy_kitchen_data/messy_kitchen_scenes_renderings"
  gt_meshes_dir: "../messy_kitchen_data/raw_messy_kitchen_scenes"

# Penetration evaluation settings (based on TAGAC implementation)
penetration:
  use_tagac_implementation: true  # Use TAGAC's well-defined penetration calculation
  voxel_size: 0.3               # Size parameter for TAGAC penetration
  voxel_resolution: 40          # Resolution parameter for TAGAC penetration
  scene_weight: 0.7             # Alpha weight for scene-level penetration
  per_object_weight: 0.3        # Beta weight for per-object penetration
  min_penetration_diff: 0.1     # Minimum difference for valid preference pairs
```

### Phase 3: Launch/Debug Configuration

#### 3.1 VS Code Launch Configuration
**File**: `.vscode/launch.json` (update existing)

**New Entries**:
```json
{
    "name": "PartCrafter DPO Data Creation",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/submodules/partcrafter_ran/scripts/create_partcrafter_dpo_data.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/submodules/partcrafter_ran",
    "env": {
        "CUDA_VISIBLE_DEVICES": "6"
    },
    "args": [
        "--input_dir", "../messy_kitchen_data/messy_kitchen_scenes_renderings",
        "--gt_dir", "../messy_kitchen_data/raw_messy_kitchen_scenes",
        "--output_dir", "dpo_data/messy_kitchen_dpo",
        "--num_candidates", "4",
        "--model_path", "pretrained_weights/PartCrafter",
        "--voxel_resolution", "0.01",
        "--min_penetration_diff", "0.1",
        "--use_tagac_penetration", "true"
    ]
},
{
    "name": "PartCrafter Penetration Score Evaluation",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/submodules/partcrafter_ran/scripts/eval_penetration_score.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/submodules/partcrafter_ran",
    "env": {
        "CUDA_VISIBLE_DEVICES": "6"
    },
    "args": [
        "--mesh_dir", "dpo_data/partcrafter_dataset",
        "--output_dir", "penetration_scores",
        "--voxel_resolution", "0.01",
        "--scene_weight", "0.7",
        "--per_object_weight", "0.3"
    ]
},
{
    "name": "PartCrafter DPO Training",
    "type": "python",
    "request": "launch",
    "module": "accelerate.commands.launch",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/submodules/partcrafter_ran",
    "env": {
        "CUDA_VISIBLE_DEVICES": "7"
    },
    "args": [
        "--num_processes=1",
        "--mixed_precision=bf16",
        "src/train_partcrafter_dpo.py",
        "--config", "configs/partcrafter_dpo.yaml"
    ]
},
{
    "name": "PartCrafter DPO Evaluation",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/submodules/partcrafter_ran/scripts/eval_partcrafter_dpo.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}/submodules/partcrafter_ran",
    "env": {
        "CUDA_VISIBLE_DEVICES": "6"
    },
    "args": [
        "--checkpoint_path", "runs/partcrafter_dpo/checkpoint-8000",
        "--eval_data_dir", "dpo_data/eval_dataset",
        "--output_dir", "results/dpo_evaluation"
    ]
}
```

### Phase 4: Evaluation and Metrics

#### 4.1 Penetration Score Evaluation
**File**: `scripts/eval_penetration_score.py` (new)

**Purpose**: Comprehensive penetration score computation for DPO constraints

##### 4.1.1 Penetration Score Definition (Based on TAGAC Implementation)

**Reference Implementation**: `submodules/TAGAC/messy_kitchen_scripts/calculate_penetration_score.py`

**Penetration Level Calculation** (from TAGAC utils):
\[
\text{PenetrationLevel} = 1 - \frac{\text{merged\_internal\_points}}{\text{individual\_internal\_points\_sum}}
\]

Where:
- `merged_internal_points`: Number of voxels inside the merged mesh
- `individual_internal_points_sum`: Sum of voxels inside each individual mesh
- **Lower score = better (less penetration), higher score = worse**

**Per-Object Penetration Analysis**:
\[
\text{PerObjectPenetration}_i = \frac{\text{overlapping\_voxels\_of\_object}_i}{\text{total\_voxels\_of\_object}_i}
\]

**DPO Constraint Score**:
\[
\text{DPOScore} = \alpha \cdot \text{PenetrationLevel} + \beta \cdot \frac{1}{N} \sum_{i=1}^{N} \text{PerObjectPenetration}_i
\]

Where:
- \(\alpha = 0.7\) (scene-level weight)
- \(\beta = 0.3\) (per-object weight)  
- \(N\) = number of objects in scene
- **Lower score = preferred, higher score = rejected**

##### 4.1.2 Implementation Details

```python
import trimesh
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Dict, Tuple

class PenetrationScoreEvaluator:
    def __init__(self, voxel_resolution: float = 0.01):
        """
        Initialize penetration score evaluator.
        
        Args:
            voxel_resolution: Resolution for voxelization (smaller = more accurate)
        """
        self.voxel_resolution = voxel_resolution
        
    def compute_scene_penetration_score(self, mesh_paths: List[str]) -> float:
        """
        Compute scene-level penetration score.
        
        Args:
            mesh_paths: List of paths to mesh files in the scene
            
        Returns:
            scene_penetration: Float between 0 and 1 (lower = better)
        """
        if len(mesh_paths) < 2:
            return 0.0  # No penetration possible with single mesh
            
        # Load all meshes
        meshes = [trimesh.load(path) for path in mesh_paths]
        
        # Voxelize each mesh
        individual_voxels = []
        for mesh in meshes:
            voxels = self._voxelize_mesh(mesh)
            individual_voxels.append(voxels)
        
        # Compute total individual voxels
        total_individual_voxels = sum(len(voxels) for voxels in individual_voxels)
        
        # Merge all meshes and voxelize
        merged_mesh = trimesh.util.concatenate(meshes)
        merged_voxels = self._voxelize_mesh(merged_mesh)
        
        # Scene penetration score
        scene_penetration = 1.0 - (len(merged_voxels) / total_individual_voxels)
        
        return scene_penetration
    
    def compute_per_object_penetration_scores(self, mesh_paths: List[str]) -> List[float]:
        """
        Compute per-object penetration scores.
        
        Args:
            mesh_paths: List of paths to mesh files in the scene
            
        Returns:
            per_object_scores: List of penetration scores for each object
        """
        if len(mesh_paths) < 2:
            return [0.0] * len(mesh_paths)
            
        # Load all meshes
        meshes = [trimesh.load(path) for path in mesh_paths]
        
        # Voxelize each mesh
        voxel_sets = [self._voxelize_mesh(mesh) for mesh in meshes]
        
        per_object_scores = []
        
        for i, voxels_i in enumerate(voxel_sets):
            if len(voxels_i) == 0:
                per_object_scores.append(0.0)
                continue
                
            # Check overlap with all other meshes
            overlapping_voxels = 0
            
            for j, voxels_j in enumerate(voxel_sets):
                if i == j:
                    continue
                    
                # Find overlapping voxels using spatial indexing
                overlap_count = self._count_overlapping_voxels(voxels_i, voxels_j)
                overlapping_voxels += overlap_count
            
            # Per-object penetration score
            penetration_score = overlapping_voxels / len(voxels_i)
            per_object_scores.append(penetration_score)
        
        return per_object_scores
    
    def compute_dpo_constraint_score(self, mesh_paths: List[str], 
                                   alpha: float = 0.7, beta: float = 0.3) -> Dict[str, float]:
        """
        Compute combined DPO constraint score.
        
        Args:
            mesh_paths: List of paths to mesh files in the scene
            alpha: Weight for scene-level penetration
            beta: Weight for per-object penetration
            
        Returns:
            scores: Dictionary containing all penetration scores
        """
        # Compute scene-level penetration
        scene_penetration = self.compute_scene_penetration_score(mesh_paths)
        
        # Compute per-object penetration
        per_object_scores = self.compute_per_object_penetration_scores(mesh_paths)
        avg_per_object_penetration = np.mean(per_object_scores) if per_object_scores else 0.0
        
        # Combined DPO score
        dpo_score = alpha * scene_penetration + beta * avg_per_object_penetration
        
        return {
            "dpo_score": dpo_score,
            "scene_penetration": scene_penetration,
            "per_object_scores": per_object_scores,
            "avg_per_object_penetration": avg_per_object_penetration
        }
    
    def _voxelize_mesh(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Voxelize a mesh into a set of voxel coordinates.
        
        Args:
            mesh: Trimesh object
            
        Returns:
            voxels: Array of voxel coordinates [N, 3]
        """
        # Get mesh bounds
        bounds = mesh.bounds
        
        # Create voxel grid
        voxel_coords = []
        
        for x in np.arange(bounds[0, 0], bounds[1, 0], self.voxel_resolution):
            for y in np.arange(bounds[0, 1], bounds[1, 1], self.voxel_resolution):
                for z in np.arange(bounds[0, 2], bounds[1, 2], self.voxel_resolution):
                    point = np.array([x, y, z])
                    
                    # Check if point is inside mesh
                    if mesh.contains([point]):
                        voxel_coords.append(point)
        
        return np.array(voxel_coords) if voxel_coords else np.empty((0, 3))
    
    def _count_overlapping_voxels(self, voxels1: np.ndarray, voxels2: np.ndarray) -> int:
        """
        Count overlapping voxels between two voxel sets.
        
        Args:
            voxels1: First voxel set [N1, 3]
            voxels2: Second voxel set [N2, 3]
            
        Returns:
            overlap_count: Number of overlapping voxels
        """
        if len(voxels1) == 0 or len(voxels2) == 0:
            return 0
            
        # Use KDTree for efficient spatial queries
        tree = cKDTree(voxels2)
        
        # Find voxels in voxels1 that are close to voxels2
        distances, _ = tree.query(voxels1)
        
        # Count voxels within voxel resolution distance
        overlap_count = np.sum(distances < self.voxel_resolution * 0.5)
        
        return overlap_count
    
    def evaluate_mesh_pair_for_dpo(self, mesh_win_path: str, mesh_loss_path: str) -> Dict[str, float]:
        """
        Evaluate a pair of meshes for DPO training.
        
        Args:
            mesh_win_path: Path to winning mesh (should have lower penetration)
            mesh_loss_path: Path to losing mesh (should have higher penetration)
            
        Returns:
            evaluation: Dictionary with penetration scores for both meshes
        """
        # Evaluate winning mesh
        win_scores = self.compute_dpo_constraint_score([mesh_win_path])
        
        # Evaluate losing mesh  
        loss_scores = self.compute_dpo_constraint_score([mesh_loss_path])
        
        return {
            "win": win_scores,
            "loss": loss_scores,
            "preference_valid": win_scores["dpo_score"] < loss_scores["dpo_score"]
        }
```

##### 4.1.3 DPO Data Generation with Penetration Constraints

```python
class PartCrafterDPODataCreator:
    def __init__(self, model_path: str, penetration_evaluator: PenetrationScoreEvaluator):
        self.pipeline = PartCrafterPipeline.from_pretrained(model_path)
        self.pipeline.cuda()
        self.penetration_evaluator = penetration_evaluator
        
    def generate_dpo_dataset_with_penetration_constraints(
        self, 
        image_paths: List[str], 
        output_dir: str,
        num_candidates: int = 4,
        min_penetration_diff: float = 0.1
    ):
        """
        Generate DPO dataset with penetration-based preference pairs.
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory for DPO data
            num_candidates: Number of candidates to generate per image
            min_penetration_diff: Minimum penetration score difference for valid pairs
        """
        for image_path in tqdm(image_paths):
            self._process_single_image(
                image_path, output_dir, num_candidates, min_penetration_diff
            )
    
    def _process_single_image(
        self, 
        image_path: str, 
        output_dir: str,
        num_candidates: int,
        min_penetration_diff: float
    ):
        """Process a single image to generate DPO data with penetration constraints."""
        
        # Generate multiple candidates
        candidates = self._generate_candidates(image_path, num_candidates)
        
        # Evaluate penetration scores for all candidates
        penetration_scores = []
        for candidate in candidates:
            scores = self.penetration_evaluator.compute_dpo_constraint_score([candidate["mesh_path"]])
            penetration_scores.append(scores["dpo_score"])
        
        # Create preference pairs based on penetration scores
        preference_pairs = self._create_penetration_based_preference_pairs(
            candidates, penetration_scores, min_penetration_diff
        )
        
        # Save DPO data
        self._save_dpo_data(image_path, candidates, penetration_scores, preference_pairs, output_dir)
    
    def _create_penetration_based_preference_pairs(
        self, 
        candidates: List[Dict], 
        penetration_scores: List[float],
        min_penetration_diff: float
    ) -> List[Dict]:
        """Create preference pairs based on penetration score differences."""
        
        preference_pairs = []
        
        # Sort candidates by penetration score (lower = better)
        sorted_indices = np.argsort(penetration_scores)
        
        # Create pairs between good and bad candidates
        for i in range(len(sorted_indices)):
            for j in range(i + 1, len(sorted_indices)):
                win_idx = sorted_indices[i]  # Lower penetration score
                loss_idx = sorted_indices[j]  # Higher penetration score
                
                penetration_diff = penetration_scores[loss_idx] - penetration_scores[win_idx]
                
                if penetration_diff >= min_penetration_diff:
                    preference_pairs.append({
                        "win_candidate": candidates[win_idx],
                        "loss_candidate": candidates[loss_idx],
                        "penetration_diff": penetration_diff,
                        "win_penetration": penetration_scores[win_idx],
                        "loss_penetration": penetration_scores[loss_idx]
                    })
        
        return preference_pairs
```

#### 4.2 Evaluation Pipeline
**File**: `scripts/eval_partcrafter_dpo.py` (new)

**Purpose**: Comprehensive evaluation of DPO-trained PartCrafter models

```python
def evaluate_dpo_model():
    # Load DPO-trained checkpoint
    model = load_partcrafter_dpo_checkpoint(checkpoint_path)
    
    # Generate meshes on evaluation dataset
    generated_meshes = generate_evaluation_meshes(model, eval_dataset)
    
    # Compute penetration scores
    penetration_scores = []
    for mesh_path in generated_meshes:
        score = penetration_evaluator.compute_penetration_score(mesh_path)
        penetration_scores.append(score)
    
    # Compute metrics
    avg_penetration = np.mean(penetration_scores)
    std_penetration = np.std(penetration_scores)
    
    # Save results
    results = {
        'average_penetration_score': avg_penetration,
        'std_penetration_score': std_penetration,
        'scores_per_mesh': penetration_scores
    }
    
    return results
```

## Implementation Steps

### Step 1: Setup and Preparation
1. **Environment Setup**
   - Ensure PartCrafter environment is properly configured
   - Install additional dependencies for DPO training (accelerate, peft, etc.)
   - Verify GPU memory requirements

2. **Data Preparation**
   - Prepare input images for DPO dataset creation
   - Set up directory structure for DPO data
   - Configure PartCrafter model paths

### Step 2: Dataset Creation
1. **Implement DPO Data Creator**
   - Adapt existing `create_dpo_data.py` for PartCrafter
   - Implement penetration score evaluation placeholder
   - Test data generation pipeline

2. **Generate DPO Dataset**
   - Run data creation script on input images
   - Verify generated preference pairs
   - Quality check on generated meshes

### Step 3: Training Implementation
1. **Adapt DPO Training Script**
   - Port DSO's DPO loss to PartCrafter architecture
   - Implement LoRA integration for efficient training
   - Test training loop with small dataset

2. **Configuration and Launch**
   - Create training configuration files
   - Update VS Code launch configurations
   - Test training startup and basic functionality

### Step 4: Evaluation and Validation
1. **Implement Evaluation Pipeline**
   - Create penetration score evaluation placeholder
   - Implement comprehensive evaluation script
   - Test evaluation on sample data

2. **Validation and Testing**
   - Run end-to-end pipeline test
   - Validate training convergence
   - Compare results with baseline PartCrafter

## File Structure Summary

```
submodules/partcrafter_ran/
├── scripts/
│   ├── create_partcrafter_dpo_data.py          # NEW: DPO dataset creation
│   ├── eval_partcrafter_dpo.py                 # NEW: DPO model evaluation
│   └── eval_penetration_score.py               # NEW: Penetration score computation
├── src/
│   ├── train_partcrafter_dpo.py                # NEW: DPO training script
│   └── data_preprocessing/
│       └── partcrafter_dpo_dataset.py          # NEW: DPO dataset loader
├── configs/
│   └── partcrafter_dpo.yaml                    # NEW: DPO training config
├── dpo_data/                                   # NEW: Generated DPO dataset
└── runs/
    └── partcrafter_dpo/                        # NEW: DPO training outputs
```

## Key Differences from DSO

1. **Model Architecture**: PartCrafter uses DiT (Diffusion Transformer) instead of TRELLIS's sparse structure flow
2. **Evaluation Metric**: Penetration score instead of physical stability
3. **Data Format**: Colorful mesh files with PartCrafter's object-centric rendering
4. **Starting Point**: PartCrafter trained checkpoints instead of TRELLIS pretrained model
5. **Pipeline**: PartCrafter's inference pipeline for data generation

## Success Metrics

1. **Training Convergence**: DPO loss decreases during training
2. **Penetration Score Improvement**: Lower average penetration scores on evaluation data
3. **Mesh Quality**: Generated meshes maintain visual quality while reducing penetration
4. **Efficiency**: Training completes within reasonable time and memory constraints

## Timeline Estimate

- **Phase 1 (Dataset Preparation)**: 2-3 days
- **Phase 2 (Training Implementation)**: 3-4 days  
- **Phase 3 (Launch Configuration)**: 1 day
- **Phase 4 (Evaluation)**: 2-3 days

**Total Estimated Time**: 8-11 days

## Penetration Score Constraints Implementation Summary

### 4.3 Concrete Implementation Plan for Penetration Constraints

#### 4.3.1 Phase 1: Penetration Score Evaluation System
1. **Implement `PenetrationScoreEvaluator` class**:
   - Scene-level penetration computation using voxelization
   - Per-object penetration analysis with spatial indexing
   - Combined DPO constraint score calculation
   - Efficient voxel overlap detection using KDTree

2. **Key Features**:
   - **Voxelization**: Convert meshes to voxel grids for overlap analysis
   - **Spatial Indexing**: Use KDTree for efficient neighbor queries
   - **Multi-scale Analysis**: Support different voxel resolutions
   - **Batch Processing**: Handle multiple meshes simultaneously

#### 4.3.2 Phase 2: DPO Data Generation with Penetration Constraints
1. **Enhanced Data Creator**:
   - Generate multiple candidates per image
   - Evaluate penetration scores for all candidates
   - Create preference pairs based on score differences
   - Filter pairs by minimum penetration difference threshold

2. **Quality Control**:
   - Minimum penetration difference: 0.1 (configurable)
   - Validation of preference pair consistency
   - Statistical analysis of score distributions

#### 4.3.3 Phase 3: Integration with Training Pipeline
1. **Dataset Integration**:
   - Load penetration scores alongside mesh data
   - Filter training samples by penetration constraints
   - Balance positive/negative preference pairs

2. **Training Modifications**:
   - Use penetration scores as preference criteria
   - Weight DPO loss by penetration score differences
   - Monitor penetration score improvements during training

#### 4.3.4 Phase 4: Evaluation and Validation
1. **Comprehensive Evaluation**:
   - Pre-training vs post-training penetration scores
   - Per-object vs scene-level improvements
   - Statistical significance testing

2. **Metrics Tracking**:
   - Average penetration score reduction
   - Percentage of improved samples
   - Distribution of score improvements

### 4.4 Expected Outcomes

#### 4.4.1 Training Improvements
- **Reduced Penetration**: Models generate meshes with lower penetration scores
- **Better Object Separation**: Improved spatial relationships between objects
- **Consistent Quality**: More reliable generation across different inputs

#### 4.4.2 Evaluation Metrics
- **Scene-Level Penetration**: Target reduction of 20-30%
- **Per-Object Penetration**: Target reduction of 15-25%
- **Combined DPO Score**: Target improvement of 25-35%

#### 4.4.3 Success Criteria
1. **Quantitative**: Measurable reduction in penetration scores
2. **Qualitative**: Visual improvement in mesh quality
3. **Consistency**: Stable improvements across diverse inputs
4. **Efficiency**: Training completes within reasonable time

## Next Steps

1. **Review and approve this implementation plan**
2. **Set up development environment and dependencies**
3. **Implement penetration score evaluation system**
4. **Begin with Phase 1: Dataset preparation flow with penetration constraints**
5. **Iteratively implement and test each phase**
6. **Validate penetration score improvements**
7. **Document results and lessons learned**
