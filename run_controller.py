#!/usr/bin/env python3
import argparse
import time

from pogs.controller import RealSenseController
from pogs.controller.policy import SimpleDepthPolicy
from pogs.controller.robot_interface import RobotInterface
import os
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interactive", "policy", "mock"], default="interactive")
    parser.add_argument("--scene_name", default="my_scene")
    parser.add_argument("--save_path", default="data/realsense_captures")
    parser.add_argument("--mock_sleep", type=float, default=0.1, help="sleep between replayed mock frames")
    args = parser.parse_args()

    rc = RealSenseController(scene_name=args.scene_name, save_path=args.save_path)
    ok = rc.connect()
    print("RealSense available:", ok)

    if args.mode == "interactive":
        p = rc.start()
        print("Launched interactive capture (pid):", p.pid if p else None)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping interactive capture...")
        finally:
            rc.stop()

    else:  # policy mode
        if args.mode == "policy":
            try:
                rc.start_stream()
            except Exception as e:
                print("Failed to start stream:", e)
                return

            frame_source = "stream"
        else:  # mock mode
            # prepare lists of saved frames under save_path/scene_name
            scene_dir = os.path.join(args.save_path, args.scene_name)
            img_dir = os.path.join(scene_dir, "img")
            depth_dir = os.path.join(scene_dir, "depth")
            if not os.path.isdir(depth_dir) or not os.path.isdir(img_dir):
                print(f"No mock data found in {scene_dir}. Ensure you have 'img/' and 'depth/' directories.")
                return
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
            if not depth_files:
                print("No depth .npy files found for mock replay.")
                return
            frame_source = "mock"
            mock_idx = 0
        try:
            rc.start_stream()
        except Exception as e:
            print("Failed to start stream:", e)
            return

        policy = SimpleDepthPolicy(target_depth=0.5)
        robot = RobotInterface()
        robot.connect()

        print("Running perception->action loop. Ctrl+C to stop.")
        try:
            while True:
                if frame_source == "stream":
                    res = rc.get_frame()
                    if res is None:
                        print("No frame received from camera")
                        time.sleep(0.1)
                        continue
                    color, depth = res
                else:
                    # load mock frames by index
                    depth_path = os.path.join(depth_dir, depth_files[mock_idx % len(depth_files)])
                    img_path = None
                    # prefer matching img filename if available
                    if mock_idx < len(img_files):
                        img_path = os.path.join(img_dir, img_files[mock_idx])
                    depth = np.load(depth_path)
                    if img_path and os.path.exists(img_path):
                        color = cv2.imread(img_path)
                    else:
                        # create a dummy color image for the policy
                        H, W = depth.shape[:2]
                        color = np.zeros((H, W, 3), dtype=np.uint8)
                    mock_idx += 1
                    time.sleep(args.mock_sleep)

                cmd = policy.propose_command(color, depth)
                print("Proposed command:", cmd)
                if cmd.type == "pose" and cmd.pose is not None:
                    # Project 6-DoF grasp to 4-DoF (fix roll/pitch) for OpenManipulator
                    from pogs.controller.robot_interface import project_pose_to_4dof
                    # Usually pi/2 (1.57) is straight down for OM-X
                    pose_4dof = project_pose_to_4dof(cmd.pose, fixed_pitch=1.57) 
                    robot.move_pose(pose_4dof)
                elif cmd.type == "joint" and cmd.joints is not None:
                    robot.move_joints(cmd.joints)
        except KeyboardInterrupt:
            print("Stopping policy loop...")
        finally:
            if args.mode == "policy":
                rc.stop_stream()


if __name__ == '__main__':
    main()
