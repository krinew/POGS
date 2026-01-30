# POGS Workflow Guide

This guide contains all commands needed to run POGS from scratch, starting with a new terminal.

## 0. Initial Setup (Run in every new terminal)

Run this command immediately after opening a terminal to configure `ninja`, `cuda`, and `nerfstudio` paths.

```bash
conda activate pogs_env
source setup_pogs_env.sh
```

---

## 1. Data Collection

Connect your Intel RealSense camera.
Change `my_scan_01` to your desired scene name.

```bash
# Capture data with 1-in-5 frame retention (reduces processing time)
python pogs/scripts/realsense_pogs_capture.py --scene_name my_scan_01 --frame_skip 5
```

*Data will be saved to:* `data/realsense_captures/my_scan_01`

---

## 2. Processing (COLMAP) - Manual Sequence

We need to compute camera poses using COLMAP. Run these 5 commands in order.

**A. Create Database Directory**
```bash
mkdir -p data/realsense_captures/my_scan_01/colmap
```

**B. Extract Features**
```bash
colmap feature_extractor \
    --database_path data/realsense_captures/my_scan_01/colmap/database.db \
    --image_path data/realsense_captures/my_scan_01/images \
    --SiftExtraction.use_gpu 1
```

**C. Match Features (Sequential with High Overlap)**
*We use `overlap 50` to help COLMAP find good initial pairs even if there are duplicate frames.*
```bash
colmap sequential_matcher \
    --database_path data/realsense_captures/my_scan_01/colmap/database.db \
    --SiftMatching.use_gpu 1 \
    --SequentialMatching.overlap 50
```

**D. Map/Reconstruct Scene**
```bash
mkdir -p data/realsense_captures/my_scan_01/colmap/sparse
colmap mapper \
    --database_path data/realsense_captures/my_scan_01/colmap/database.db \
    --image_path data/realsense_captures/my_scan_01/images \
    --output_path data/realsense_captures/my_scan_01/colmap/sparse \
    --Mapper.ba_global_function_tolerance=1e-6
```

**E. Convert to Nerfstudio Format**
*We use `ns-process-data` but skip the actual COLMAP steps since we just did them.*
```bash
# Note: Ensure your sparse output is in colmap/sparse/0 before running this.
# If colmap created 'sparse/0', verify it with: ls data/realsense_captures/my_scan_01/colmap/sparse/0
ns-process-data images \
    --data data/realsense_captures/my_scan_01/images \
    --output-dir data/realsense_captures/my_scan_01 \
    --skip-colmap \
    --matching-method sequential
```

---

## 3. Training

Train the Gaussian Splatting model.
*Note: If you get a compilation error, verify you ran `source setup_pogs_env.sh`.*

```bash
# Clear compilation cache if you suspect build issues (optional, implies recompilation)
# rm -rf ~/.cache/torch_extensions/

# Start Training (WITH DEPTH)
# --depths-path tells it to use the 'depth' folder we captured
ns-train pogs \
    --data data/realsense_captures/my_scan_01 \
    colmap \
    --depths-path depth
```

*Wait for training to complete (approx 4000 steps).*
*Note the output path at the end, e.g., `outputs/my_scan_01/pogs/2026-01-25_223408/config.yml`*

---

## 4. Visualization & Clustering

Launch the viewer using the config file generated in Step 3.

```bash
# Replace with YOUR actual config path from Step 3
ns-viewer --load-config outputs/my_scan_01/pogs/YYYY-MM-DD_HHMMSS/config.yml
```

**In the Browser:**
1. Go to the URL shown in the terminal (e.g., `http://localhost:7007`).
2. **Cluster Objects**:
   - Move camera to look at an object.
   - Click **"Crop to Click"** (Right panel).
   - Click the object in the 3D scene.
   - Click **"Add Crop to Group"**.
