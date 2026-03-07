## POGS + OMX 4-DOF Grasp Execution

Integrated pipeline for OpenManipulator X 4-DOF control using POGS grasp net and LeRobot IK.

### Files

**1. `pogs/grasping/omx_grasp_executor.py`** (294 lines)
- `OMXGraspExecutor` class: Bridge between grasp net and robot control
- `project_pose_to_4dof()`: Constrains 6-DOF grasps to 4-DOF (roll=0, yaw-free)
- Dual IK: Pinocchio (if available) + Geometric fallback
- Methods:
  - `grasp_to_joints()`: Convert single 6-DOF grasp to joint angles
  - `batch_grasps()`: Convert N grasps with success mask
  - `execute_grasp()`: Direct robot execution

**2. `scripts/demo_omx_grasp_execution.py`** (190 lines)
- End-to-end demo: POGS → Grasp Net → 4-DOF → IK → Robot
- Loads grasps from grasp net or mock data
- Filters by IK feasibility
- Executes top-K grasps by confidence
- Saves results (poses + joint configs)

### Quick Start

```bash
# Dry-run with mock grasps
python scripts/demo_omx_grasp_execution.py --dry-run --top-k 3

# Real execution (requires calibrated robot)
python scripts/demo_omx_grasp_execution.py --scene data/my_scene --output outputs/my_scene

# Custom pitch (e.g., grasp from above: -π/2)
python scripts/demo_omx_grasp_execution.py --dry-run --pitch -1.57
```

### Pipeline

```
[POGS Tracking + Segmentation]
             ↓
[Grasp Net: 6-DOF Grasps (N, 4, 4)]
             ↓
[4-DOF Projection: fix roll/pitch]
             ↓
[IK Solver (Pinocchio or Geometric)]
             ↓
[Joint Commands (N, 4)]
             ↓
[OMX Robot Execution]
```

### Key Features

- **Respect OMX Constraints**: 4-DOF (yaw, x, y, z) only
- **Automatic IK**: Falls back to geometric solver if Pinocchio unavailable
- **Batch Processing**: Convert multiple grasps efficiently
- **Dry-Run Mode**: Test without hardware
- **Minimal Dependencies**: Uses only LeRobot + scipy + numpy

### Integration with Existing Code

#### Use OMXGraspExecutor with POGS pipeline:

```python
from pogs.grasping.omx_grasp_executor import OMXGraspExecutor
from pogs.grasping.generate_grasps_ply import generate_grasps

# Generate 6-DOF grasps from grasp net
grasps_6dof, scores, contact_pts = generate_grasps(...)

# Create executor
executor = OMXGraspExecutor(robot=robot_instance)

# Convert and execute
for grasp, score in zip(grasps_6dof, scores):
    success = executor.execute_grasp(grasp, fixed_pitch=-1.57)
    if success:
        print(f"Grasp executed! (score={score:.2f})")
```

#### Or use batch processing:

```python
joints, mask = executor.batch_grasps(grasps_6dof, fixed_pitch=-1.57)

# Filter feasible grasps
feasible_idx = np.where(mask)[0]
feasible_joints = joints[feasible_idx]
```

### Notes

- **Pitch Angle**: 
  - `1.57` = Downward (gripper pointing down)
  - `-1.57` = Upward (gripper pointing up)
  - `0.0` = Horizontal
  
- **IK Fallback**: Geometric IK is always available; Pinocchio is optional for better accuracy
- **OMX Specs**: L1=0.077m, L2=0.13m, L3=0.124m, L4=0.126m (link lengths)
- **Gripper**: Controlled separately via normalized commands (0-100)
