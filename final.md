## scene capture

```bash
conda activate pogs_env

export COPPELIASIM_ROOT=/home/pi0/CoppeliaSim
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export PYTHONPATH=$PYTHONPATH:/home/pi0/POGS-ACT-implementation/POGS/PointCloudMatters

# 2. Run the script natively! A real CoppeliaSim window will pop up!
cd /home/pi0/POGS-ACT-implementation/POGS/POGS
python pogs/scripts/capture_rlbench_scene.py \
    --task        open_drawer \
    --episode     0 \
    --data-root   /home/pi0/POGS-ACT-implementation/POGS/PointCloudMatters/data/rlbench/raw/train \
    --out-dir     data/pogs_scenes/open_drawer/shared \
    --n-views     100
```

## train

```bash
conda activate pogs_env

# 1. Completely nuke the polluted library paths that are causing the crash
unset LD_LIBRARY_PATH
unset LIBRARY_PATH

# 2. Re-add ONLY the pure POGS environment libraries
export POGS_ENV_ROOT="/home/pi0/miniconda3/envs/pogs_env"
export LD_LIBRARY_PATH="$POGS_ENV_ROOT/lib"
```

### Note

If training has an out of dimension error then you have to basically clear the dino cache:
#### Delete the outdated 5-frame DINO cache
```bash
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/dino.*
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/*.npy
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/*.info

# Delete the cached CLIP directory
rm -rf /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/clip_*
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/detic.npy
```

#### Restart the training! It will now extract features for all 600 frames.
```bash
ns-train pogs --data data/pogs_scenes/open_drawer/shared
```

## After training, extract the point cloud 

```bash
mkdir -p data/pogs_act/open_drawer

# 2. Extract the Gaussian point cloud!
python pogs/scripts/export_pogs_pointcloud.py \
    --checkpoint outputs/shared/pogs/try/nerfstudio_models/step-000003000.ckpt \
    --out data/pogs_act/open_drawer/shared.ply \
    --max-points 8192
```

---

# UPDATED COMMANDS -> YEH FINAL HAIIIII

To view the nerf after training:
```bash
ns-viewer --load-config outputs/shared/pogs/2026-04-03_010833/config.yml
```

### 4. Final command for training with updated parameters
```bash
ns-train pogs --data data/pogs_scenes/open_drawer/shared \
    --pipeline.model.densify-grad-thresh 0.002 \
    --pipeline.model.cull-alpha-thresh 0.05 \
    --pipeline.model.stop-split-at 2500 \
    --pipeline.model.cull-screen-size 0.15 \
    --pipeline.model.cull-scale-thresh 0.5
```

---

## Final Dataset Generation

Once you have trained the model and used `ns-viewer` to crop and track the dynamic object (e.g., the drawer), you can generate the dataset.

```bash
# 5. Generate the dataset natively
python pogs/scripts/generate_pogs_act_dataset.py \
  --task open_drawer \
  --raw-root /home/pi0/POGS-ACT-implementation/POGS/PointCloudMatters/data/rlbench/raw/train \
  --pogs-config outputs/shared/pogs/2026-04-03_010833/config.yml \
  --out-dir exports/act_datasets/ \
  --pointnet2-ckpt /home/pi0/POGS-ACT-implementation/POGS/Pointnet_Pointnet2_pytorch/log/part_seg/pointnet2_part_seg_msg/checkpoints/best_model.pth
```

### Important Notes on Dataset Generation:
* **Headless Background Engine**: The dataset generation uses the CoppeliaSim background engine to replay RLBench episodes, but runs entirely **headless**. No visual GUI will pop up on your desktop!
* **Full Context Rendering (`clusters.npy`)**: When using the viewer to crop the object, `clusters.npy` properly records just `1` transform tracking group for the dynamic object (e.g. the drawer). This is intended behavior! POGS only needs to track what actually moves.
* **Whole Scene Embeddings**: Even though only the drawer is structurally "tracked" by the optimizer, POGS maintains the full >1.18M background points as static context. The entire visual scene (static environment + tracked moving drawer) gets dynamically reconstructed, scaled up to 500x500 to match the optimizer, re-rendered, and seamlessly embedded by PointNet++!

### Steps to take note of

1. **Initialization and Loading**
   * **Dataset Path Resolution**: The script first searches the directory you provided (`--raw-root .../raw/train`) to find all saved RLBench episodes for the specific task (`open_drawer`).
   * **POGS Model Checkpoint**: It reads the `config.yml` from your `--pogs-config` argument, which tells nerfstudio the dimensions of the pretrained model and where to find the `.ckpt` parameters for Gaussian Splatting.
   * **PointNet++ Instantiation**: We initialize the empty `PointNet2Encoder` onto the GPU with `in_channels=135`.

2. **The Episode Simulation Loop (Offline Reconstruction)**
   * For each raw trajectory it finds (e.g., `episode0.pkl`, `episode1.pkl`):
     * **Reset State**: The POGS Optimizer resets the rigid object translations back to the starting pose.
     * **Frame Propagation**: We iterate over every single timestep (T) in that episode's camera recording.
     * **POGS Refinement**: For each new RGB-D image in the sequence (frame T), the POGS Optimizer executes 5 gradient steps (`--niters=5`). It adjusts the cam2world transformation, rigidly tracks the objects frame-to-frame, and renders out the updated Gaussian cloud. Note: RLBench records at 128x128 but we cleanly upscale this natively to 500x500 for POGS gradient rendering.

3. **Masking & Semantic Feature Interleaving**
   * Now the real magic happens for each frame T:
     * **Extraction**: Once POGS tracks the frame, we pull the raw means (3D XYZ), colors (RGB), dino_feats (128-dim), detic_feats (1-dim), and your clustered masks (clusters.npy labels natively pinned to each point) out of the optimizer's memory stack.
     * **Preprocessing**: The colors are normalized exactly in accordance with the PointCloudMatters codebase requirements (from 0-255 down to `[-1, 1]`).
     * **Feature Tensor Assembly**: Everything is zipped together into one flat tensor array: `[colors, coords, dino, detic, clusters]`, yielding exactly 135 combined feature channels per point.

4. **PointNet++ Embedding Generation**
   * **Network Pass**: The entire 135-channel point cloud for frame T is forwarded straight through the `PointNet2Encoder`.
   * **Dimension Condensation**: PointNet++ processes the global context and spits out exactly one 1024-float embedding vector representing that precise timestep's entire semantic state!

5. **Packaging The Episode**
   * Finally, when the loop reaches the end of the episode (e.g., all 60 frames have been track-embedded):
     * It sweeps up the robot's state (`joint_positions`, `gripper_open`).
     * It tracks the ground-truth action the robotic arm executed.
     * It constructs the payload dictionary. Our custom-injected `obs_embeds: (T, 1024)` sits right at the top.
     * It outputs `exports/act_datasets/episode0.pkl`, which acts as the ultimate master file that your ACT policy script requires!


cache=1189336, model=1183023