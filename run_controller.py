#!/usr/bin/env python3
import argparse
import time

from pogs.controller import RealSenseController
from pogs.controller.robot_interface import RobotInterface
import os
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interactive"], default="interactive")
    parser.add_argument("--scene_name", default="my_scene")
    parser.add_argument("--save_path", default="data/realsense_captures")
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


if __name__ == '__main__':
    main()
