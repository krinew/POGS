"""
Demo script for POGS Pick-and-Place pipeline (Mock Mode).
Simulates the full flow: Scene -> Object -> Grasping -> Planning -> Execution.
"""

import numpy as np
import logging
import sys
import os
from unittest.mock import MagicMock

# Mock torch if not available (for lightweight demo)
try:
    import torch
except ImportError:
    print("[DEMO] Torch not found, using Mock...")
    import types
    torch_mock = types.ModuleType('torch')
    torch_mock.Tensor = MagicMock()
    sys.modules['torch'] = torch_mock
    
    # Mock trimesh as it is used in ToadObject
    # Instead of mocking trimesh submodules, let's mock the consumer
    toad_mock_module = MagicMock()
    # Ensure 'from pogs.tracking.toad_object import ToadObject' works
    toad_mock_module.ToadObject = MagicMock
    sys.modules['pogs.tracking.toad_object'] = toad_mock_module

    # Mock open3d just in case
    sys.modules['open3d'] = MagicMock()
    sys.modules['open3d.visualization'] = MagicMock()
    
    # Mock pogs.tracking.toad_object to intercept ToadObject import if needed
    # But PickAndPlaceController imports it directly.
    # If ToadObject is imported, it will try to use trimesh.
    # By mocking trimesh above, we might be fine importing the real file if it doesn't do too much at module level.
    # Let's see if we can get away with just mocking trimesh.
    
    # Mock POGS pipeline to avoid deep dependency hell (nerfstudio/cuda/etc)
    # DO NOT MOCK 'pogs' package itself, or we lose file traversal for other modules
    
    # We still need pogs.manipulation.pick_and_place to load
    # so we can't fully mock 'pogs' if we want to import submodules naturally
    # but we can populate sys.modules so 'from pogs.pogs_pipeline import ...' works
    
    # Let's just mock the specific heavy module
    pp_mock = MagicMock()
    sys.modules['pogs.pogs_pipeline'] = pp_mock
    
    # Also mock tracking
    # We need to be careful with ToadObject if we import it later
    # sys.modules['pogs.tracking.toad_object'] = MagicMock() <- This might break imports
    # Better to mock the problematic internal imports of tracking modules
    sys.modules['pogs.tracking.rigid_group_optimizer'] = MagicMock()
    
    # Mock Controller commands if needed
    # (commands.py is lightweight usually, maybe fine)
    
    # Re-enable torch basic stuff if needed by other modules
    sys.modules['torch'].eye = np.eye  # dummy 
    
    try:
        import scipy
    except ImportError:
        print("[DEMO] Scipy not found, mocking...")
        scipy_mock = MagicMock()
        sys.modules['scipy'] = scipy_mock
        sys.modules['scipy.spatial'] = MagicMock()
        sys.modules['scipy.spatial.transform'] = MagicMock()
        
        # We need to fake Rotation for robot_interface to load
        class MockRotation:
            @staticmethod
            def from_matrix(m): return MockRotation()
            @staticmethod
            def from_euler(s, a): return MockRotation()
            def as_euler(self, s): return [0, 0, 0]
            def as_matrix(self): return np.eye(3)
        
        sys.modules['scipy.spatial.transform'].Rotation = MockRotation
        pass


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pogs.manipulation.pick_and_place import PickAndPlaceController
from pogs.controller.robot_interface import RobotInterface
from pogs.tracking.toad_object import ToadObject
from pogs.manipulation.grasp_utils import project_6dof_to_4dof_grasps

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DEMO] - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("DEMO")

class MockPOGSPipeline:
    def __init__(self):
        self.model = MagicMock()
        # Mock clustering
        self.model.cluster_labels = None # Simulate not computed yet or mock it?
    
    def get_object_point_cloud(self):
        # Return a dummy cloud: a cube centered at [0.4, 0.0, 0.05]
        points = np.random.rand(100, 3) * 0.1 # 10cm box
        points[:, 0] += 0.4
        points[:, 2] += 0.05
        return points

class MockRealSenseController:
    pass

class MockPickAndPlace(PickAndPlaceController):
    """Subclass to inject mock behaviors for parts that require full GPU/Hardware."""
    
    def capture_and_build_scene(self):
        logger.info("[Mock] Capturing scene... done.")
        return np.zeros((100, 3)) # Dummy cloud

    def segment_and_extract_object(self, object_query: str):
        logger.info(f"[Mock] Segmenting '{object_query}' (DETIC + POGS)... found.")
        
        # Create a dummy object (mug-sized cylinder)
        points = np.random.rand(500, 3) * 0.1
        points[:, 0] += 0.4 # 40cm forward
        points[:, 2] += 0.1 # 10cm up
        
        # We return a simple struct/mock similar to ToadObject
        obj = MagicMock(spec=ToadObject)
        obj.points = points
        obj.scene_scale = 1.0
        return obj

    def generate_grasps(self, obj, grasp_checkpoint_dir):
        logger.info("[Mock] Running ContactGraspNet (Inference)...")
        
        # Generate some random 6-DOF grasps around the object centroid
        obj_center = np.mean(obj.points, axis=0) # [0.45, 0.05, 0.15] approximately
        
        grasps = []
        scores = []
        
        center_pose = np.eye(4)
        center_pose[:3, 3] = obj_center
        
        # 1. Good top-down grasp (Pitch ~ 90 deg)
        # We'll make one that is perfect for our 4-DOF filtering
        # Rotation: Roll=0, Pitch=1.57, Yaw=0
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler('xyz', [0, 1.57, 0])
        g1 = center_pose.copy()
        g1[:3, :3] = r.as_matrix()
        grasps.append(g1)
        scores.append(0.95) # High score

        # 2. Impossible side grasp (Pitch ~ 0)
        # Might be filtered out or projected heavily depending on logic
        r2 = R.from_euler('xyz', [0, 0, 0])
        g2 = center_pose.copy()
        g2[:3, :3] = r2.as_matrix()
        grasps.append(g2)
        scores.append(0.80)
        
        # 3. Low score grasp
        grasps.append(g1.copy())
        scores.append(0.10)
        
        return np.array(grasps), np.array(scores)

    def _get_object_pose_from_pogs(self):
        return np.eye(4)

def run_demo():
    print("="*60)
    print("POGS 4-DOF Pick-and-Place DEMO")
    print("="*60)
    
    # 1. Initialize Mock Components
    robot = RobotInterface() # Simulates Dry-Run if no robot connected
    
    # 2. Instantiate Controller
    controller = MockPickAndPlace(
        pogs_pipeline=MockPOGSPipeline(),
        camera=MockRealSenseController(),
        robot=robot,
        config={
            "grasp_height_above_table": 0.05,
            "lift_height": 0.15,
            "approach_distance": 0.1,
            "gripper_open_value": 1.0,
            "gripper_closed_value": 0.0,
            "gripper_move_time": 0.5,
            "max_grasps_to_consider": 5,
            "grasp_score_threshold": 0.5,
        }
    )
    
    # 3. Run Pipeline
    target_object = "mug"
    place_target = np.eye(4)
    place_target[:3, 3] = [0.2, -0.3, 0.2] # Place to the side
    
    success = controller.run_pick_and_place(
        object_query=target_object,
        grasp_checkpoint_dir="dummy/path",
        place_location=place_target
    )
    
    if success:
        print("\n\033[92m[SUCCESS] Demo completed successfully!\033[0m")
    else:
        print("\n\033[91m[FAILURE] Demo failed.\033[0m")

if __name__ == "__main__":
    run_demo()
