# Controller Module

The `pogs/controller` module provides a unified interface for perception→action control: it ingests camera frames (RGB + depth) from a RealSense camera, runs a policy to compute motor commands, and sends commands to a robot (e.g., UR5 via `ur5py`).

## Architecture

### Core Components

- **`ControllerBase`** (`controller.py`): Abstract interface defining `connect()`, `disconnect()`, `start()`, `stop()`.
- **`RealSenseController`** (`controller.py`): Concrete implementation wrapping Intel RealSense D435/D455 cameras.
  - Interactive mode: spawns the interactive capture script as a subprocess.
  - Stream mode: in-process pipeline for programmatic frame access.
- **`Command`** (`commands.py`): Dataclass representing motor commands (joint angles or pose matrices).
- **`SimpleDepthPolicy`** (`policy.py`): Simple perception→action policy that computes pose deltas based on mean depth in image center.
- **`RobotInterface`** (`robot_interface.py`): Thin wrapper for sending commands to UR5 robot (graceful dry-run if hardware unavailable).

### Usage Flow

```
RealSenseController.start_stream()
    ↓
RealSenseController.get_frame()  →  (color, depth) numpy arrays
    ↓
SimpleDepthPolicy.propose_command(color, depth)  →  Command
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

### 2. Policy Mode (Live Control)

Run the perception→action loop in real-time using RealSense frames:

```bash
python3 run_controller.py --mode policy \
  --scene_name my_scene \
  --save_path data/realsense_captures
```

The policy will:
1. Capture frames from the RealSense pipeline.
2. Compute a Cartesian pose delta based on mean depth (target: 0.5 m).
3. Propose and send the command to the robot (or dry-run if hardware unavailable).
4. Loop at ~10 Hz.

**Press Ctrl+C** to stop.

### 3. Mock Mode (Offline Testing)

Replay saved frames (from a prior capture) and run the policy without hardware:

```bash
python3 run_controller.py --mode mock \
  --scene_name my_scan_01 \
  --save_path outputs/my_scan_01/pogs \
  --mock_sleep 0.1
```

This mode:
1. Loads depth `.npy` files and color `.png` files from `<save_path>/<scene_name>/depth/` and `<save_path>/<scene_name>/img/`.
2. Cycles through saved frames at the rate specified by `--mock_sleep` (seconds).
3. Runs the same policy on each frame.
4. Outputs proposed commands (dry-run mode if robot unavailable).

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

### SimpleDepthPolicy

```python
from pogs.controller.policy import SimpleDepthPolicy
import numpy as np

policy = SimpleDepthPolicy(target_depth=0.5, crop_frac=0.2, z_gain=0.5)

# Propose a command given color and depth frames
cmd = policy.propose_command(color, depth)
print(cmd.type, cmd.joints, cmd.pose)
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
├── policy.py                # SimpleDepthPolicy
├── robot_interface.py       # RobotInterface
└── README.md                # This file
```

## Examples

### Example 1: Run Policy in Real-Time

```python
from pogs.controller import RealSenseController
from pogs.controller.policy import SimpleDepthPolicy
from pogs.controller.robot_interface import RobotInterface
import time

rc = RealSenseController()
if not rc.connect():
    print("RealSense not available")
    exit(1)

rc.start_stream()
policy = SimpleDepthPolicy(target_depth=0.5)
robot = RobotInterface()
robot.connect()

try:
    for _ in range(100):
        color, depth = rc.get_frame()
        cmd = policy.propose_command(color, depth)
        if cmd.type == "pose":
            robot.move_pose(cmd.pose)
        time.sleep(0.1)
finally:
    rc.stop_stream()
```

### Example 2: Replay Saved Frames

```python
import os
import numpy as np
import cv2
from pogs.controller.policy import SimpleDepthPolicy
from pogs.controller.robot_interface import RobotInterface

depth_dir = "outputs/my_scan_01/pogs/depth"
img_dir = "outputs/my_scan_01/pogs/img"

depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
policy = SimpleDepthPolicy()
robot = RobotInterface()
robot.connect()

for depth_file in depth_files[:10]:
    depth = np.load(os.path.join(depth_dir, depth_file))
    color = cv2.imread(os.path.join(img_dir, depth_file.replace('.npy', '.png')))
    cmd = policy.propose_command(color, depth)
    print(f"Frame {depth_file}: {cmd}")
    if cmd.type == "pose":
        robot.move_pose(cmd.pose)
```

## Extending the Controller

To implement a custom policy, subclass `SimpleDepthPolicy` or create a new class with a `propose_command(color, depth) -> Command` method:

```python
from pogs.controller.policy import SimpleDepthPolicy
from pogs.controller.commands import Command
import numpy as np

class CustomPolicy(SimpleDepthPolicy):
    def propose_command(self, color, depth):
        # Your perception logic here
        # e.g., run a neural network on color+depth
        # return Command(type="pose", pose=...) or Command(type="joint", joints=...)
        pass
```

Then pass your custom policy to the controller loop in `run_controller.py`.

## Troubleshooting

- **"RealSense available: False"**: RealSense library (`pyrealsense2`) not installed or no device detected. Try `pip install pyrealsense2` or check USB connection.
- **"No depth .npy files found"**: Mock mode couldn't find saved frames. Ensure `--save_path/--scene_name/depth/` directory exists with `.npy` files.
- **Robot connection fails**: `ur5py` not installed or UR5 not reachable on network. Commands will run in dry-run mode.

## License

Part of the POGS project. See top-level `LICENSE`.
