# PartCrafter DPO Implementation Summary

## Overview
This document provides a high-level summary of implementing Direct Preference Optimization (DPO) finetuning for PartCrafter using penetration score as the evaluation metric. The goal is to improve PartCrafter's 3D mesh generation quality by training on preference data based on penetration analysis.

### Data Sources
- **Input Images**: `messy_kitchen_data/messy_kitchen_scenes_renderings/` - Contains rendering images as model inputs
- **Ground Truth Meshes**: `messy_kitchen_data/raw_messy_kitchen_scenes/` - Contains GLB mesh files for evaluation
- **Penetration Score**: Reference implementation in `submodules/TAGAC/messy_kitchen_scripts/calculate_penetration_score.py`

## Core Concept

### DPO Training Approach
DPO (Direct Preference Optimization) is a method for aligning generative models with human preferences. Instead of training on absolute quality scores, DPO learns from relative preferences between pairs of outputs.

### Penetration Score as Quality Metric
- **Lower penetration score = better quality** (less object overlap)
- **Higher penetration score = worse quality** (more object overlap)
- Uses TAGAC's well-defined penetration level calculation

### Key Mathematical Formula
**Penetration Level**:
```
PenetrationLevel = 1 - (merged_internal_points / individual_internal_points_sum)
```

**DPO Constraint Score**:
```
DPOScore = α × PenetrationLevel + β × Average(PerObjectPenetration)
```
Where α=0.7 (scene-level weight), β=0.3 (per-object weight)

## Implementation Phases

### Phase 1: Dataset Preparation Flow

#### 1.1 DPO Dataset Creation
**Purpose**: Generate diverse 3D mesh outputs using PartCrafter with different random seeds from messy_kitchen_data, then evaluate them for penetration scores using TAGAC's well-defined penetration level calculation.

**Data Flow**:
1. **Input**: `messy_kitchen_scenes_renderings/` (rendering images)
2. **Generation**: PartCrafter with different seeds → multiple mesh candidates
3. **Evaluation**: TAGAC penetration score calculation
4. **Output**: DPO preference pairs based on penetration scores

**Key Components**:
- PartCrafterDPODataCreator class
- Penetration score evaluation using TAGAC implementation
- Preference pair creation based on penetration scores
- Integration with messy_kitchen_data structure

**Directory Structure**:
```
dpo_data/messy_kitchen_dpo/
├── scene_id_combined/
│   ├── 000/  # seed 0
│   │   ├── pred_merged.glb
│   │   ├── pred_obj_*.ply  # individual meshes
│   │   ├── penetration_analysis.json
│   │   ├── seed_info.json
│   │   └── latent_features.pt
│   ├── 001/  # seed 1, 002/  # seed 2, 003/  # seed 3
│   ├── input_image.jpg  # from rendering
|   |── Image Features   
│   ├── gt_mesh.glb      # from raw_messy_kitchen_scenes
│   ├── preference_pairs.json
│   └── scene_metadata.json
```
```
dpo_data/messy_kitchen_dpo/
├── scene_id_combined/
│   ├── 000/  # seed 0
│   │   ├── pred_merged.glb
│   │   ├── pred_obj_*.ply  
│   │   ├── penetration_analysis.json
│   │   ├── seed_info.json
│   │   └── latent_features.pt
│   ├── 001/  # seed 1, 002/  # seed 2, 003/  
│   ├── input_image.jpg  
|   |── Image Features   
│   ├── gt_mesh.glb    
│   ├── preference_pairs.json
│   └── scene_metadata.json
```

**Data Mapping**:
- **Input Images**: `messy_kitchen_scenes_renderings/{scene_id}_combined/rendering.png`
- **Ground Truth**: `raw_messy_kitchen_scenes/{scene_id}_combined.glb`
- **Scene Info**: `messy_kitchen_scenes_renderings/{scene_id}_combined/num_parts.json`, `iou.json`

#### 1.2 Data Processing Pipeline
**Purpose**: Process DPO data into training format, adapted from DSO's dataset.py for PartCrafter architecture

**Key Components**:
- PartCrafterDPODataset class
- Multiview sampling support
- Penetration threshold filtering
- Latent feature extraction

**Data Structure Differences**:
- **DSO**: Uses sparse structure tensors directly
- **PartCrafter**: Uses VAE-encoded latent features from point clouds
- **Evaluation**: Penetration scores instead of stability angles

### Phase 2: DPO Finetuning Training Script

#### 2.1 Main Training Script
**File**: `src/train_partcrafter_dpo.py`

**Based on**: DSO's `finetune.py` + PartCrafter's `train_partcrafter.py`

#### 2.2 Key Modifications from DSO to PartCrafter

**Model Loading**:
- Replace DSO's TRELLIS model with PartCrafter DiT model
- Load TripoSG VAE, DINOv2 image encoder, and BitImageProcessor
- Use trained PartCrafter checkpoints as starting point

**Pipeline Adaptation**:
- Replace Trellis ImageTo3DPipeline with PartCrafterPipeline
- Adapt data loading for PartCrafter's object-centric rendering

**Dataset Integration**:
- Replace SyntheticDataset with PartCrafterDPODataset
- Change stability_threshold to penetration_threshold
- Update batch keys from sparse_x0 to latent_win/loss

**DPO Loss Function**:
- Adapt forward_dpo_loss for PartCrafter transformer architecture
- Use latent features instead of sparse structures
- Implement PartCrafter-specific noise prediction

**LoRA Configuration**:
- Update target_modules for PartCrafter DiT attention layers
- Configure for PartCrafter's transformer architecture

**Configuration Changes**:
- Model paths: PartCrafter checkpoints instead of TRELLIS
- Dataset settings: penetration_threshold instead of stable_threshold
- Data format: latent_win/loss tensors instead of sparse_x0

### Phase 3: Launch/Debug Configuration

#### 3.1 VS Code Launch Configuration
**New Entries**:
- PartCrafter DPO Data Creation
- PartCrafter Penetration Score Evaluation  
- PartCrafter DPO Training
- PartCrafter DPO Evaluation

#### 3.2 Configuration Files
**File**: `configs/partcrafter_dpo.yaml`

**Key Settings**:
- Model paths for PartCrafter components
- DPO hyperparameters (beta, sample_same_epsilon)
- LoRA configuration for efficient training
- Dataset paths for messy_kitchen_data
- Penetration evaluation parameters

### Phase 4: Evaluation and Metrics

#### 4.1 Penetration Score Evaluation
**Reference Implementation**: TAGAC's `calculate_penetration_score.py`

**Key Features**:
- Scene-level penetration analysis
- Per-object penetration computation
- Combined DPO constraint scoring
- Integration with messy_kitchen_data

**Evaluation Metrics**:
- Average penetration score reduction
- Percentage of improved samples
- Distribution of score improvements
- Statistical significance testing

#### 4.2 Expected Outcomes

**Training Improvements**:
- Reduced penetration in generated meshes
- Better object separation and spatial relationships
- More consistent generation quality across inputs

**Evaluation Metrics**:
- Scene-Level Penetration: Target reduction of 20-30%
- Per-Object Penetration: Target reduction of 15-25%
- Combined DPO Score: Target improvement of 25-35%

**Success Criteria**:
- Quantitative: Measurable reduction in penetration scores
- Qualitative: Visual improvement in mesh quality
- Consistency: Stable improvements across diverse inputs
- Efficiency: Training completes within reasonable time

## Implementation Steps

### Step 1: Setup and Preparation
1. **Environment Setup**
   - Configure PartCrafter environment
   - Install DPO training dependencies (accelerate, peft, etc.)
   - Verify GPU memory requirements

2. **Data Preparation**
   - Prepare messy_kitchen_data for DPO dataset creation
   - Set up directory structure for DPO data
   - Configure PartCrafter model paths

### Step 2: Dataset Creation
1. **Implement DPO Data Creator**
   - Adapt existing create_dpo_data.py for PartCrafter
   - Integrate TAGAC penetration score evaluation
   - Test data generation pipeline

2. **Generate DPO Dataset**
   - Run data creation script on messy_kitchen_data
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
   - Create penetration score evaluation using TAGAC implementation
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
│   └── datasets/
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
6. **Data Source**: messy_kitchen_data with TAGAC penetration evaluation

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

## Next Steps

1. **Review and approve this implementation summary**
2. **Set up development environment and dependencies**
3. **Begin with Phase 1: Dataset preparation flow**
4. **Iteratively implement and test each phase**
5. **Validate penetration score improvements**
6. **Document results and lessons learned**

## Technical Notes

### Penetration Score Constraints Implementation
- Use TAGAC's well-defined penetration level calculation
- Integrate scene-level and per-object penetration analysis
- Implement efficient voxel overlap detection
- Support batch processing for multiple meshes

### DPO Data Generation with Penetration Constraints
- Generate multiple candidates per image using different seeds
- Evaluate penetration scores for all candidates
- Create preference pairs based on score differences
- Filter pairs by minimum penetration difference threshold

### Training Pipeline Integration
- Load penetration scores alongside mesh data
- Filter training samples by penetration constraints
- Balance positive/negative preference pairs
- Weight DPO loss by penetration score differences
- Monitor penetration score improvements during training

### Comprehensive Evaluation
- Pre-training vs post-training penetration scores
- Per-object vs scene-level improvements
- Statistical significance testing
- Average penetration score reduction tracking
- Percentage of improved samples analysis
