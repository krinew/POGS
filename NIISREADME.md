
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
