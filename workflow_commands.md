# POGS-ACT Dynamic Scene Pipeline Workflow

This document outlines the step-by-step commands to run the newly refactored dynamic POGS-ACT pipeline. All steps assume you are running from the workspace root (`/home/pi0/POGS-ACT-implementation`).

## 0. Environment Setup
Before running any scripts, assure your Conda environment is activated:
```bash
source ~/miniconda3/bin/activate
conda activate pogs_env
```

## 1. Dataset Generation (Raw Episode Replay)
Extract dynamic point clouds and POGS tracking data from raw RLBench episodes by replaying them step-by-step.

```bash
rm -rf data/pogs_scenes/open_drawer/shared
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/dino.*
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/.npy 
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/.info
rm -rf /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/clip_*
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/detic.npy
```


```bash
python POGS/pogs/scripts/generate_pogs_act_dataset.py \
    --task open_drawer \
    --raw-root PointCloudMatters/data/rlbench/raw/train \
    --pogs-config POGS/POGS/outputs/shared/pogs/2026-03-30_221559/config.yml \
    --out-dir PointCloudMatters/data/rlbench/pogs_extracted/train
```

## 2. Precompute PointNet++ Embeddings
Run the frozen PointNet++ encoder offline to convert the tracked point clouds into 1024-dimensional embeddings. Doing this offline avoids massive compute overhead during ACT training.
```bash
python PointCloudMatters/scripts/precompute_pogs_pointnet2_embeddings.py \
    --input-dir PointCloudMatters/data/rlbench/pogs_extracted/train \
    --out-dir PointCloudMatters/data/rlbench/pogs_embeddings/train
```

## 3. Train the ACT Policy
Train the Action Chunking with Transformers (ACT) model using the newly created dataset containing precomputed `obs_embeds`.
```bash
python PointCloudMatters/src/train.py \
    experiment=pogs_pointnet2_act \
    data.dataset_dir=PointCloudMatters/data/rlbench/pogs_embeddings/train
```
*(Note: Adjust the `experiment` flag and dataset paths depending on the exact hydra config structure you are using)*

## 4. Live Inference / End-to-End Evaluation
Run the standalone end-to-end RLBench test script. This script runs RLBench closed-loop, performing live POGS tracking, PointNet++ encoding, and ACT inference at every timestep.
```bash
python POGS/pogs/scripts/live_pogs_inference.py \
    --task open_drawer \
    --checkpoint PointCloudMatters/logs/pogs_pointnet2_act/checkpoints/last.ckpt
```