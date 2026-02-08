# Controller Module

The `pogs/controller` module provides a unified interface for perception→action control: it ingests camera frames (RGB + depth) from a RealSense camera and sends commands to a robot (e.g., UR5 via `ur5py`).

## Architecture

### Core Components

- **`ControllerBase`** (`controller.py`): Abstract interface defining `connect()`, `disconnect()`, `start()`, `stop()`.
- **`RealSenseController`** (`controller.py`): Concrete implementation wrapping Intel RealSense D435/D455 cameras.
  - Interactive mode: spawns the interactive capture script as a subprocess.
  - Stream mode: in-process pipeline for programmatic frame access.
- **`Command`** (`commands.py`): Dataclass representing motor commands (joint angles or pose matrices).
- **`RobotInterface`** (`robot_interface.py`): Thin wrapper for sending commands to UR5 robot (graceful dry-run if hardware unavailable).

### Usage Flow

```
RealSenseController.start_stream()
    ↓
RealSenseController.get_frame()  →  (color, depth) numpy arrays
    ↓
RobotInterface.move_pose() or move_joints()
```

## Installation

Ensure the `pogs_env` conda environment is active and the core dependencies are installed:

```bash
conda activate pogs_env
```

## Usage

### 1. Interactive Mode

Launch the interactive RealSense capture GUI (record frames manually with spacebar):

```bash
python3 run_controller.py --mode interactive \
  --scene_name my_scene \
  --save_path data/realsense_captures
```

**Controls** (in the capture window):
- **Space**: toggle recording on/off
- **s**: save a single frame (manual trigger)
- **q** or **ESC**: quit

### 2. Online Control (POGS)

POGS handles online decision-making and control externally. This module focuses on camera I/O and robot command transport, not a built-in policy.

## API Reference

### RealSenseController

```python
from pogs.controller import RealSenseController

rc = RealSenseController(scene_name="my_scene", save_path="data/realsense_captures")

# Check if RealSense is available
ok = rc.connect()
print(f"RealSense available: {ok}")

# Start in-process streaming (not interactive)
rc.start_stream(width=1280, height=720, fps=30)

# Get a frame
color, depth = rc.get_frame(timeout_ms=5000)
# color: HxWx3 uint8 BGR
# depth: HxW float32 (meters)

# Stop streaming
rc.stop_stream()

# Disconnect
rc.disconnect()
```

### Command

```python
from pogs.controller.commands import Command

# Create a pose command
cmd = Command(type="pose", pose=pose_matrix.tolist())

# Create a joint command
cmd = Command(type="joint", joints=[0.0, -1.57, 1.57, -1.57, 0.0, 0.0])

# Convert to JSON
json_str = cmd.to_json()
```

### RobotInterface

```python
from pogs.controller.robot_interface import RobotInterface

robot = RobotInterface()
ok = robot.connect()  # Try to connect to UR5; returns False if unavailable
print(f"Robot connected: {ok}")

# Send a pose command (4x4 matrix)
pose = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]]
robot.move_pose(pose, vel=0.3, acc=0.1)

# Send joint angles
robot.move_joints([0.0, -1.57, 1.57, -1.57, 0.0, 0.0], vel=0.3, acc=0.1)
```

## Directory Structure

```
pogs/controller/
├── __init__.py              # Exports ControllerBase, RealSenseController
├── commands.py              # Command dataclass
├── controller.py            # ControllerBase and RealSenseController
├── robot_interface.py       # RobotInterface
└── README.md                # This file
```

## Examples

### Example 1: Stream Frames and Send Commands

```python
from pogs.controller import RealSenseController
from pogs.controller.robot_interface import RobotInterface
import time

rc = RealSenseController()
if not rc.connect():
    print("RealSense not available")
    exit(1)

rc.start_stream()
robot = RobotInterface()
robot.connect()

try:
    for _ in range(100):
        color, depth = rc.get_frame()
        # Compute or retrieve a command from your online POGS loop
        # cmd = ...
        # robot.move_pose(cmd.pose) or robot.move_joints(cmd.joints)
        time.sleep(0.1)
finally:
    rc.stop_stream()
```

### Example 2: Replay Saved Frames

```python
import os
import numpy as np
import cv2
from pogs.controller.robot_interface import RobotInterface

depth_dir = "outputs/my_scan_01/pogs/depth"
img_dir = "outputs/my_scan_01/pogs/img"

depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
robot = RobotInterface()
robot.connect()

for depth_file in depth_files[:10]:
    depth = np.load(os.path.join(depth_dir, depth_file))
    color = cv2.imread(os.path.join(img_dir, depth_file.replace('.npy', '.png')))
    # Compute or retrieve a command from your online POGS loop
    # cmd = ...
    # robot.move_pose(cmd.pose) or robot.move_joints(cmd.joints)
```

## Extending the Controller

Implement your online POGS loop externally and call `RobotInterface.move_pose()` or `RobotInterface.move_joints()` with the commands it produces.

## Troubleshooting

- **"RealSense available: False"**: RealSense library (`pyrealsense2`) not installed or no device detected. Try `pip install pyrealsense2` or check USB connection.
- **Robot connection fails**: `ur5py` not installed or UR5 not reachable on network. Commands will run in dry-run mode.

## License

Part of the POGS project. See top-level `LICENSE`.
