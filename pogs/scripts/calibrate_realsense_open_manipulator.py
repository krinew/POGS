import sys
import os
import time
from pathlib import Path

# Add project root to sys.path to allow imports from pogs package
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))

import numpy as np
import cv2
import tyro
from autolab_core import RigidTransform
from pogs.tracking.realsense_wrapper import RealSense
from pogs.controller.open_manipulator import OpenManipulatorRobot


def _detect_aruco_pose(rgb, K, dist, tag_length, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return None

    obj_points = np.array([
        [-tag_length / 2, tag_length / 2, 0],
        [ tag_length / 2, tag_length / 2, 0],
        [ tag_length / 2,-tag_length / 2, 0],
        [-tag_length / 2,-tag_length / 2, 0]
    ], dtype=np.float32)

    img_points = corners[0].reshape((4, 2))
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not success:
        return None

    R_cam_tag, _ = cv2.Rodrigues(rvec)
    t_cam_tag = tvec.reshape(3)

    # cam_from_tag: p_cam = R * p_tag + t
    cam_from_tag = np.eye(4)
    cam_from_tag[:3, :3] = R_cam_tag
    cam_from_tag[:3, 3] = t_cam_tag

    return cam_from_tag


def _pose_wrapper_to_matrix(pose):
    if hasattr(pose, "matrix"):
        return pose.matrix
    return np.array(pose)


def main(
    tag_length: float = 0.05,
    num_samples: int = 15,
    aruco_dict: str = "DICT_4X4_50",
    output_name: str = "world_to_extrinsic_zed_for_grasping_down.tf",
    wrist_to_tag_path: Path | None = None,
    resolution: str = "720p",
    fps: int = 30,
):
    """
    Calibrate a fixed RealSense camera to the robot/world frame using an ArUco tag
    rigidly attached to the robot wrist.

    The output is a camera->world transform saved under calibration_outputs.
    """

    aruco_dict_type = getattr(cv2.aruco, aruco_dict, cv2.aruco.DICT_4X4_50)

    camera = RealSense(flip_mode=False, resolution=resolution, fps=fps)
    robot = OpenManipulatorRobot(gripper=True)

    if wrist_to_tag_path is not None and wrist_to_tag_path.exists():
        wrist_to_tag = RigidTransform.load(str(wrist_to_tag_path))
    else:
        wrist_to_tag = RigidTransform()
        wrist_to_tag.from_frame = "tag"
        wrist_to_tag.to_frame = "wrist"
        print("[WARN] Using identity wrist_to_tag. For accurate results, provide --wrist_to_tag_path.")

    K = camera.get_K()
    dist = camera.get_distortion()

    cam_to_world_rvecs = []
    cam_to_world_tvecs = []

    print("\nCalibration started.")
    print("Attach ArUco tag to the wrist, move robot to varied poses.")
    print("Press Enter to capture each sample, or type 'q' then Enter to finish.\n")

    captured = 0
    while captured < num_samples:
        user_in = input(f"Capture sample {captured+1}/{num_samples} (Enter/q): ")
        if user_in.strip().lower() == "q":
            break

        rgb, _, _ = camera.get_frame(depth=True)
        if rgb is None:
            print("[WARN] No frame captured.")
            continue

        rgb = rgb.detach().cpu().numpy()
        cam_from_tag = _detect_aruco_pose(rgb, K, dist, tag_length, aruco_dict_type)
        if cam_from_tag is None:
            print("[WARN] No ArUco detected.")
            continue

        pose = robot.get_pose()
        world_from_wrist = _pose_wrapper_to_matrix(pose)

        # world_from_cam = world_from_wrist * wrist_from_tag * tag_from_cam
        wrist_from_tag = wrist_to_tag.matrix if hasattr(wrist_to_tag, "matrix") else np.array(wrist_to_tag)
        tag_from_cam = np.linalg.inv(cam_from_tag)
        world_from_cam = world_from_wrist @ wrist_from_tag @ tag_from_cam

        rvec, _ = cv2.Rodrigues(world_from_cam[:3, :3])
        tvec = world_from_cam[:3, 3].reshape(3, 1)

        cam_to_world_rvecs.append(rvec)
        cam_to_world_tvecs.append(tvec)
        captured += 1
        print(f"[OK] Captured sample {captured}")

    if len(cam_to_world_rvecs) == 0:
        print("No samples captured. Exiting.")
        return

    mean_rvec = np.mean(np.array(cam_to_world_rvecs), axis=0)
    mean_tvec = np.mean(np.array(cam_to_world_tvecs), axis=0)

    mean_R, _ = cv2.Rodrigues(mean_rvec)
    mean_t = mean_tvec.reshape(3)

    cam_to_world = RigidTransform(
        rotation=mean_R,
        translation=mean_t,
        from_frame="camera",
        to_frame="world",
    )

    calibration_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../calibration_outputs")
    os.makedirs(calibration_save_path, exist_ok=True)

    output_path = os.path.join(calibration_save_path, output_name)
    cam_to_world.save(output_path)

    print(f"\nSaved camera->world transform to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
