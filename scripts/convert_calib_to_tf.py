import json
import numpy as np
from autolab_core import RigidTransform

# Load JSON
with open('pogs/calibration_outputs/world_to_extrinsic_realsense.json', 'r') as f:
    T_cam_base_list = json.load(f)

T_cam_base = np.array(T_cam_base_list)

# Inverse logic? Wait, the tracking script wants world->extrinsic (which is camera frame)
# If T_cam_base is the transform of base in cam, then T_base_cam is cam in base.
# It depends on how POGS uses it. Usually it wants Camera in World frame (T_world_cam).
# Hand-Eye Output gives T_cam_base (usually T_cam2gripper = Eye-in-Hand, T_cam2base = Eye-to-Hand)
# Let's save it directly as RigidiTransform for autolab_core.

rt = RigidTransform(
    rotation=T_cam_base[:3, :3],
    translation=T_cam_base[:3, 3],
    from_frame='realsense_extrinsic',
    to_frame='world'
)

rt.save('pogs/calibration_outputs/world_to_extrinsic_realsense.tf')
print("Successfully generated world_to_extrinsic_realsense.tf")
