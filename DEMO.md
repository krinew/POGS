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

## 5) Provide a valid POGS config

What to do:
- Train or load a POGS scene and locate the config.yml in outputs.
- Update the path in track_main_online_demo.py (config_path parameter) to the correct config.yml.

## 6) Run the demo with hardware

Command:
python3 pogs/scripts/track_main_online_demo.py

What it does: runs the full pipeline with RealSense + OpenManipulator.

## 7) Viewer URL

What to do:
- Open the URL printed by Viser (default: http://0.0.0.0:8080).

## Notes

- If you see “No device connected,” the RealSense isn’t detected.
- If you see “could not open port /dev/ttyUSB0,” the robot serial port is missing or different.
- If the config.yml path is wrong, Optimizer init will fail.
