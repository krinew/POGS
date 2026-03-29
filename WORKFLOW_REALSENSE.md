# POGS Workflow for Handheld RealSense (Official + Adapted)

This workflow is adapted from the official POGS instructions to work with a handheld Intel RealSense camera instead of a robot arm.

## 1. Setup Environment
Ensure your environment is set up and active.
```bash
conda activate pogs_env
# Set necessary environment variables for every session
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## 2. Capture Data
We use a specific script for RealSense capture because the official `scene_capture.py` requires a robot connection.

**Camera Tips:**
- **Move Slowly:** Avoid motion blur.
- **Translate:** Move the camera around the object, do not just rotate in place.
- **Coverage:** Get about 100-200 frames.

```bash
# Replace 'my_scan_03' with your desired scene name
python pogs/scripts/realsense_pogs_capture.py --scene_name my_scan_03 --frame_skip 1
```

## 3. Process Data (COLMAP)
The official POGS workflow assumes the robot provides poses. Since we are handheld, we must calculate poses using `nerfstudio`'s processing tool (which runs COLMAP).

**Note:** We use `--matching-method exhaustive` for better reconstruction results on small object scans.

```bash
# This generates the required transforms.json with correct poses
ns-process-data images \
    --data data/realsense_captures/my_scan_03/images \
    --output-dir data/realsense_captures/my_scan_03 \
    --matching-method exhaustive
```

*Verify success: Check that `data/realsense_captures/my_scan_03/transforms.json` exists and contains camera frames.*

## 4. Train POGS
Now we run the official training command, pointing to our processed data.
We explicitly add `--depths-path depth` to utilize the depth maps captured by the RealSense.

```bash
ns-train pogs \
    --data data/realsense_captures/my_scan_03 \
    --depths-path depth
```

## 5. View & Interact
After training finishes, the viewer will launch automatically (or you can verify using the command below).
Follow the "Cluster Scene" steps from the official README in the viewer:
1.  **Toggle RGB/Cluster** button.
2.  **Cluster Scene** button.
3.  **Click** on the object -> **Crop to Click**.
4.  **Add Crop to Group List**.

```bash
# If you need to re-open the viewer later (replace timestamp with your actual folder)
ns-viewer --load-config outputs/my_scan_03/pogs/YYYY-MM-DD_HHMMSS/config.yml
```
