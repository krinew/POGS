## scene capture

```
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

```
conda activate pogs_env

# 1. Completely nuke the polluted library paths that are causing the crash
unset LD_LIBRARY_PATH
unset LIBRARY_PATH

# 2. Re-add ONLY the pure POGS environment libraries
export POGS_ENV_ROOT="/home/pi0/miniconda3/envs/pogs_env"
export LD_LIBRARY_PATH="$POGS_ENV_ROOT/lib"

# 3. Train the scene!
cd /home/pi0/POGS-ACT-implementation/POGS/POGS
ns-train pogs --data data/pogs_scenes/open_drawer/shared

```
### Note

If training has out of dimension error then you have to basically cler the dino cache 
#### Delete the outdated 5-frame DINO cache
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/dino.*

rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/*.npy
rm -f /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/*.info
# Delete the cached CLIP directory
rm -rf /home/pi0/POGS-ACT-implementation/POGS/POGS/outputs/shared/clip_*

#### Restart the training! It will now extract features for all 600 frames.
ns-train pogs --data data/pogs_scenes/open_drawer/shared

##After this we have to extract the point cloud 

```
mkdir -p data/pogs_act/open_drawer
# 2. Extract the Gaussian point cloud!
python pogs/scripts/export_pogs_pointcloud.py \
    --checkpoint outputs/shared/pogs/try/nerfstudio_models/step-000003000.ckpt \
    --out data/pogs_act/open_drawer/shared.ply \
    --max-points 8192

```



