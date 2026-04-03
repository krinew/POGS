# POGS-ACT Implementation: Pipeline & Bug Fixes Changelog

This document catalogs all the modifications made across the POGS pipeline to stabilize the conversion of RLBench Simulation data into tracked 3DGS offline datasets for ACT policy training. The primary issues addressed were 24GB RTX 4090 VRAM limitations (OOM errors), Viser web UI crashes, and simulation coordinate axis mismatches.

## 1. Tracking Script Fixes (`pogs/scripts/generate_pogs_act_dataset.py`)
- **Axis Coordinate Flipping:** Altered the `init_cam_pose` extraction logic. NeRF / OpenGL camera coordinates flip the Y and Z axes relative to RLBench. We patched the tracking transformation matrix to properly flip the components to align the simulated camera with the NeRF world space.
- **Dynamic VRAM Freezing:** Tracking massive 64D semantic networks (DINO & CLIP) iteratively across an entire episode instantly blew up VRAM. We froze the semantic arrays using `model.dino_feats.requires_grad_(False)` and `model.nn_projection.requires_grad_(False)` so the `RigidGroupOptimizer` only spends memory calculating physical rigid offsets (Translations/Rotations) over time.

## 2. Core Splatting & Clustering Fixes (`pogs/pogs.py`)
- **HDBSCAN Clustering OOM Fix (Voxel Downsampling):** The original logic passed 1.18M–1.6M points straight into HDBSCAN, causing a deadly `rmm::out_of_memory` C++ 2GB allocation crash. 
  - *Fix:* Increased Open3D's `voxel_down_sample_and_trace` voxel size parameter from `0.0001` (1/10th of a mm) to `0.05` (5cm). This safely thins the point cloud before clustering to prevent WebSocket disconnections and VRAM spikes.
  - *Note:* The simulation bounding box geometry `np.clip(..., -1, 1)` was restored to maintain the original pipeline logic per request.
- **Removed Blocking `input()` Calls:** When point clouds exceeded 1,000,000 points, the code utilized `input("Are you sure? y/n")`. Because Viser operates in an asynchronous WebSocket loop, standard Python terminal blocking breaks the UI pings, causing the browser to timeout and crash. The prompt was bypassed.

## 3. Web UI & Interaction Fixes (`pogs/pogs_pipeline.py`)
- **Rayclick Out-Of-Bounds Fix:** Clicking points on the screen edge triggered an `IndexError: index 989 is out of bounds for axis 0 with size 500`. Fixed by strictly clamping the raycast indexing coordinates: `pix_y = max(0, min(pix_y, max_y - 1))` and `pix_x = max(0, min(pix_x, max_x - 1))`.
- **"Device-Side Assert NaN" Crop Crash Fix:** When clicking "Crop bounds" on the drawer, a hardcoded RLBench table filter attempted to violently delete any point below `table_z_val`. Because axes rotate from Simulation to NeRF mathematically, this filter mistakenly deleted *all* points in the drawer cluster. Passing an empty 0-dimensional tensor into the CUDA rasterizer caused the engine to freeze.
  - *Fix:* Injected safety checks around `filtered_inds`. If the table filter reduces the cluster array size to `0`, it ignores the filter instead to ensure the CUDA renderer never crashes gracefully.

## 4. Tyro Training CLI Arguments (VRAM Optimizations)
Without manually enforcing densification caps, default Gaussian Splatting eagerly spawns over 1.6 Million points. When the `lerf_step` (semantic network) engages midway through the run, it instantly throws an Out-Of-Memory error. Required strict training flags were codified to prevent unchecked growth:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ns-train pogs ... \
  --pipeline.model.stop-split-at 2500 \
  --pipeline.model.densify-grad-thresh 0.002 \
  --pipeline.model.cull-alpha-thresh 0.05
```

## 5. Environment Alignment
Fixed an issue where `pip` was mapped to an older legacy source code directory `/home/pi0/POGS_NEW/POGS`. Force-reinstalled the active workspace via `pip install -e .` from within `/home/pi0/POGS-ACT-implementation/POGS/POGS` to ensure active Python changes manifest in terminal executions natively.
- **April 3, 2026**: Fixed `device-side assert` during `Crop to Click` causing crashes. Synchronized `self.model.cluster_labels` index arrays to reduce dynamically with `self.model.gauss_params`, preventing massive out of bounds exceptions when evaluating semantic grouping colors on physically cropped scenes.
- **April 3, 2026**: Fixed `device-side assert` during `Crop to Click` causing crashes. Synchronized `self.model.cluster_labels` index arrays to proportionally reduce with `self.model.gauss_params`, preventing massive out of bounds exceptions when evaluating semantic grouping colors on physically cropped scenes.
