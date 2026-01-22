import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Capture RGB and Aligned Depth data from Intel RealSense for POGS.")
    parser.add_argument("--scene_name", type=str, default="my_scene", help="Name of the scene (creates a folder with this name).")
    parser.add_argument("--save_path", type=str, default="data/realsense_captures", help="Base path to save data.")
    return parser.parse_args()

def setup_directories(base_path, scene_name):
    # Mimic the POGS scene_capture.py directory structure
    scene_dir = os.path.join(base_path, scene_name)
    dirs = {
        "img": os.path.join(scene_dir, "img"),
        "depth": os.path.join(scene_dir, "depth"),
        "depth_png": os.path.join(scene_dir, "depth_png"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def main():
    args = parse_args()
    save_dirs = setup_directories(args.save_path, args.scene_name)
    
    print(f"Saving data to: {os.path.join(args.save_path, args.scene_name)}")

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    print(f"Found RealSense device: {device_product_line}")

    # Set resolution - HD is usually good for POGS/Nerfstudio
    W, H = 1280, 720
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than clipping_distance_in_meters away
    clipping_distance_in_meters = 2.0 # 2 meter clipping
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    print("\n--- Controls ---")
    print("Press [Space] to toggle recording ON/OFF.")
    print("Press [s] to save a single frame (if not recording).")
    print("Press [q] or [ESC] to quit.")
    
    frame_count = 0
    recording = False
    
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # Depth in meters = depth_image * depth_scale
            depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert depth to meters (float32) for POGS
            depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale

            # Render images:
            #   depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)
            #   We can use a simpler viz
            depth_image_viz = cv2.normalize(depth_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_image_viz, cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Add Info Text
            cv2.putText(images, f"Recording: {recording} | Frames: {frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if recording else (0, 0, 255), 2)
            cv2.imshow('RealSense Capture (Color + Depth)', images)
            
            key = cv2.waitKey(1)
            
            save_frame = False
            # Check keys
            if key & 0xFF == ord('q') or key == 27:
                break
            elif key & 0xFF == ord(' '):
                recording = not recording
                print(f"Recording state: {recording}")
            elif key & 0xFF == ord('s'):
                save_frame = True # Manual single frame capture
            
            if recording or save_frame:
                # Format filenames 
                # POGS: frame_00000.png, frame_00000.npy
                
                # Increment frame count
                frame_idx = frame_count
                frame_count += 1
                
                # Save Paths
                img_path = os.path.join(save_dirs['img'], f"frame_{frame_idx:05d}.png")
                depth_npy_path = os.path.join(save_dirs['depth'], f"frame_{frame_idx:05d}.npy")
                depth_png_path = os.path.join(save_dirs['depth_png'], f"frame_{frame_idx:05d}.png")
                
                # Save Color
                cv2.imwrite(img_path, color_image) # cv2 saves as BGR match rs.format.bgr8
                
                # Save Depth NPY (Meters)
                np.save(depth_npy_path, depth_image_meters)
                
                # Save Depth PNG (Viz)
                # POGS scene_capture uses matplotlib jet, let's stick to cv2 for speed or simple generic
                cv2.imwrite(depth_png_path, depth_colormap)
                
                if save_frame:
                    print(f"Saved frame {frame_idx}")

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Captured {frame_count} frames to {os.path.join(args.save_path, args.scene_name)}")

if __name__ == "__main__":
    main()
