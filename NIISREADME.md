
Activates the pogs_env and runs the track_main_onlinedemo.py


source /home/pi0/miniconda3/etc/profile.d/conda.sh && conda activate pogs_env && cd /home/pi0/POGS && python3 pogs/scripts/track_main_online_demo.py

For port not found (in case terminal prompts lerobot find port)

use this -- (make port accessible)

sudo chmod 666 /dev/ttyUSB0 && ls -la /dev/ttyUSB0

---

## GraspNet → Robot Pipeline Demo

This demo tests the full grasp pipeline:
1. Generate mock grasps (simulating GraspNet output)
2. Use LeRobot OpenManipulator control
3. Execute pick-and-place sequence on real robot

### Prerequisites

- Robot connected via USB (`/dev/ttyUSB0`)
- Robot motors are Leader IDs: 21-25 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper)

### Commands

**1. Set port permissions (if needed):**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**2. Run basic robot movement test:**
```bash
source /home/pi0/miniconda3/etc/profile.d/conda.sh && conda activate pogs_env && cd /home/pi0/POGS && echo "2" | python3 scripts/test_robot_movement.py
```

**3. Run GraspNet → Robot pipeline test:**
```bash
source /home/pi0/miniconda3/etc/profile.d/conda.sh && conda activate pogs_env && cd /home/pi0/POGS && python3 scripts/test_graspnet_to_robot.py
```

### What the GraspNet Demo Does

1. **Connects to robot** via LeRobot DynamixelMotorsBus (IDs 21-25)
2. **Generates mock grasps** simulating GraspNet output (joint-space targets)
3. **Selects best grasp** based on score
4. **Executes grasp sequence:**
   - Move to home position
   - Open gripper
   - Move to pre-grasp position (approach)
   - Move to grasp position
   - Close gripper (grasp object)
   - Retract (lift up)
   - Move to place position (opposite side)
   - Open gripper (release object)
   - Return to home

### Test Scripts

| Script | Description |
|--------|-------------|
| `scripts/test_robot_movement.py` | Basic robot movement test (SDK test, gripper, joints) |
| `scripts/test_graspnet_to_robot.py` | Full GraspNet → Robot pipeline demo |

### Robot Configuration

- **Motor IDs:** Leader arm (21-25)
  - 21: shoulder_pan (degrees)
  - 22: shoulder_lift (normalized -100 to 100)
  - 23: elbow_flex (normalized -100 to 100)
  - 24: wrist_flex (normalized -100 to 100)
  - 25: gripper (0=closed, 100=open)

### Troubleshooting

**Port not accessible:**
```bash
sudo chmod 666 /dev/ttyUSB0
```

**Motor not detected:**
```bash
# Scan for motors using Dynamixel SDK
source /home/pi0/miniconda3/etc/profile.d/conda.sh && conda activate pogs_env && python3 -c "
from dynamixel_sdk import *
port = PortHandler('/dev/ttyUSB0')
packet = PacketHandler(2.0)
port.openPort()
port.setBaudRate(1000000)
for i in range(1, 30):
    model, comm, _ = packet.ping(port, i)
    if comm == 0: print(f'ID {i}: Model {model}')
port.closePort()
"
```

**LeRobot OmxFollower error (wrong IDs):**
- Your robot uses Leader IDs (21-25), not Follower IDs (1-5)
- Use `OpenManipulatorLeRobot` with `use_leader_ids=True`

---

## ✅ POGS + ContactGraspNet Integration (Completed 7 March 2026)

Successfully integrated POGS scene reconstruction with ContactGraspNet for real robot grasping.

### What Works

1. **ContactGraspNet Environment** - Created `contact_graspnet_env` with TensorFlow 2.5, Python 3.8
2. **TF Ops Compiled** - PointNet++ sampling/grouping/interpolation ops built for CUDA
3. **Checkpoints Downloaded** - `scene_test_2048_bs3_hor_sigma_001` model ready
4. **Point Cloud Export** - Export PLY from trained POGS model (148,709 gaussians)
5. **Grasp Generation** - ContactGraspNet generates 90 grasps from scene
6. **Robot Execution** - Grasps execute on OpenManipulator X via LeRobot

### Commands

**Export point cloud and generate grasps:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate pogs_env && \
unset LD_LIBRARY_PATH && export LD_LIBRARY_PATH=/home/pi0/miniconda3/envs/pogs_env/lib:$LD_LIBRARY_PATH && \
python scripts/export_and_grasp.py --config outputs/my_scene/pogs/2026-01-25_223408/config.yml
```

**Execute grasp on robot (dry run):**
```bash
python scripts/execute_omx_grasp.py --grasp-dir outputs/my_scene/pogs/2026-01-25_223408/grasps --dry-run
```

**Execute grasp on robot (real):**
```bash
python scripts/execute_omx_grasp.py --grasp-dir outputs/my_scene/pogs/2026-01-25_223408/grasps --execute
```

### New Scripts Created

| Script | Description |
|--------|-------------|
| `scripts/export_and_grasp.py` | Export POGS point cloud → Run ContactGraspNet |
| `scripts/execute_omx_grasp.py` | Transform grasps → IK → Execute on OMX robot |
| `pogs/dependencies/contact_graspnet/contact_graspnet/inference_no_viz.py` | ContactGraspNet inference without mayavi/pyrender |

### Key Fixes Applied

1. **cuDNN conflict** - Set `LD_LIBRARY_PATH=/home/pi0/miniconda3/envs/pogs_env/lib` to avoid asr_env cuDNN
2. **YAML loader** - Fixed `yaml.load()` to use `Loader=yaml.SafeLoader`
3. **Point cloud scaling** - Scale POGS scene (arbitrary units) to ContactGraspNet range (~0.5m)
4. **Coordinate transform** - Camera frame → Robot base frame mapping

### ⚠️ Known Issues

1. **Calibration needed** - Robot-to-camera transform is placeholder. Run:
   ```bash
   python pogs/scripts/calibrate_realsense_open_manipulator.py
   ```
   Requires ArUco tag on robot wrist.

2. **Coordinate mapping** - Current `execute_omx_grasp.py` uses approximate transform.
   After calibration, update to use proper `world_to_extrinsic_zed_for_grasping_down.tf`

### Files Generated

```
outputs/my_scene/pogs/2026-01-25_223408/
├── scene.ply                 # Exported point cloud (148,709 points)
├── best_grasp.npy            # Best grasp pose (4x4 SE3)
└── grasps/
    ├── grasp_poses.npy       # 90 grasp poses
    ├── grasp_scores.npy      # Grasp confidence scores (0.18-0.31)
    └── grasp_contacts.npy    # Contact points
```

### Environment Details

| Environment | Python | Key Packages |
|-------------|--------|--------------|
| `pogs_env` | 3.10 | PyTorch, nerfstudio, open3d |
| `contact_graspnet_env` | 3.8 | TensorFlow 2.5, CUDA 11.2, cudnn 8.2 |
