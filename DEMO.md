# POGS Demo Checklist (RealSense + OpenManipulator)

This guide lists what you must do to run the RealSense + OpenManipulator demo end‑to‑end, plus the commands and what they do.

## 1) Activate environment

Command:
conda activate pogs_env

What it does: activates the POGS Conda environment with required Python deps.

## 2) (Optional) Dry‑run compile check (no hardware)

Command:
python3 pogs/scripts/track_main_online_demo.py --dry-run

What it does: starts the demo with dummy camera + dummy robot. This only checks imports and basic setup.

## 3) Connect hardware

What to do:
- Plug in the RealSense camera (USB 3.0).
- Plug in the OpenManipulator controller and confirm the serial device (typically /dev/ttyUSB0).

## 4) Calibrate camera → world (RealSense + OpenManipulator)

Command:
python3 pogs/scripts/calibrate_realsense_open_manipulator.py --tag-length 0.05

What it does: captures multiple ArUco poses with the tag on the wrist and saves camera→world to calibration_outputs/world_to_extrinsic_zed_for_grasping_down.tf.

Optional (if you have a known wrist→tag transform):
python3 pogs/scripts/calibrate_realsense_open_manipulator.py --tag-length 0.05 --wrist-to-tag-path /path/to/wrist_to_tag.tf

What it does: uses the known wrist→tag transform for higher accuracy.

## 5) Capture Scene Data (RealSense)

Command:
```bash
python pogs/scripts/realsense_pogs_capture.py --scene_name my_scan_03 --frame_skip 1
```

What it does: captures RGB + Depth images from RealSense for POGS training.

Controls:
- **SPACE** - Toggle recording
- **S** - Save single frame
- **Q** - Quit

Tips:
- Move the camera slowly around the object (avoid motion blur).
- Translate the camera, don't just rotate in place.
- Capture 100-200 frames for good coverage.

## 6) Process Data with COLMAP

Command:
```bash
ns-process-data images \
    --data data/realsense_captures/my_scan_03/images \
    --output-dir data/realsense_captures/my_scan_03 \
    --matching-method exhaustive
```

What it does: runs COLMAP to compute camera poses from the captured images.

Verify: check that `data/realsense_captures/my_scan_03/transforms.json` exists and contains camera frames.

## 7) Train POGS

Command:
```bash
ns-train pogs \
    --data data/realsense_captures/my_scan_03 \
    --depths-path depth
```

What it does: trains the POGS model on your captured scene. The viewer will launch automatically after training.

In the viewer:
1. Click **Toggle RGB/Cluster** button.
2. Click **Cluster Scene** button.
3. Click on the object → **Crop to Click**.
4. Click **Add Crop to Group List**.

## 8) Provide a valid POGS config

What to do:
- After training, locate the config.yml in `outputs/my_scan_03/pogs/YYYY-MM-DD_HHMMSS/config.yml`.
- Update the path in track_main_online_demo.py (config_path parameter) to the correct config.yml.

To re-open the viewer later:
```bash
ns-viewer --load-config outputs/my_scan_03/pogs/YYYY-MM-DD_HHMMSS/config.yml
```

## 9) Run the demo with hardware

Command:
python3 pogs/scripts/track_main_online_demo.py

What it does: runs the full pipeline with RealSense + OpenManipulator.

## 10) Viewer URL

What to do:
- Open the URL printed by Viser (default: http://0.0.0.0:8080).

## Notes

- If you see “No device connected,” the RealSense isn’t detected.
- If you see “could not open port /dev/ttyUSB0,” the robot serial port is missing or different.
- If the config.yml path is wrong, Optimizer init will fail.
