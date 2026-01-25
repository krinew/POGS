import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
import json
import time
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser(description="Capture RGB, Depth, and camera poses from Intel RealSense for POGS training.")
    parser.add_argument("--scene_name", type=str, default="my_realsense_scene", help="Name of the scene (creates a folder with this name).")
    parser.add_argument("--save_path", type=str, default="data/realsense_captures", help="Base path to save data.")
    parser.add_argument("--frame_skip", type=int, default=1, help="Only save every Nth frame during recording (1=save all, 2=save every other frame, etc.).")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS (default: 30).")
    return parser.parse_args()

def setup_directories(base_path, scene_name):
    scene_dir = os.path.join(base_path, scene_name)
    dirs = {
        "color": os.path.join(scene_dir, "images"), # Nerfstudio usually expects 'images'
        "depth": os.path.join(scene_dir, "depth"),
        "depth_npy": os.path.join(scene_dir, "depth_npy"), # POGS often uses raw npy depth
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return scene_dir, dirs

def get_intrinsics_dict(profile):
    intr = profile.as_video_stream_profile().get_intrinsics()
    return {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "distortion_params": intr.coeffs,
        "model": "OPENCV" # Realsense uses Brown-Conrady which is compatible with OPENCV in Nerfstudio
    }

def main():
    args = parse_args()
    scene_dir, save_dirs = setup_directories(args.save_path, args.scene_name)
    
    print(f"Saving data to: {scene_dir}")
    print("Controls:\n  [SPACE] Toggle Recording\n  [S] Save Single Frame\n  [Q] Quit")

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams (HD resolution is standard for POGS/Nerfstudio)
    W, H = 1280, 720
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, args.fps)

    # Start streaming
    profile = pipeline.start(config)

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get intrinsics for transforms.json
    color_stream = profile.get_stream(rs.stream.color)
    intrinsics = get_intrinsics_dict(color_stream)

    recording = False
    frame_count = 0
    frames_data = [] # To store list of frames for transforms.json
    start_time = time.time()
    frame_counter = 0  # Counter for frame skipping

    try:
        while True:
            # Wait for coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Normalize depth for visualization (optional)
            depth_image_viz = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_image_viz, cv2.COLORMAP_JET)

            # Show images
            images = np.hstack((color_image, depth_colormap))
            
            # Overlay status
            status_color = (0, 255, 0) if recording else (0, 0, 255)
            status_text = "RECORDING" if recording else "PAUSED"
            skip_text = f" | Skip: {args.frame_skip}" if args.frame_skip > 1 else ""
            cv2.putText(images, f"{status_text} | Frames: {frame_count}{skip_text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('RealSense Capture (Color + Depth)', images)
            
            key = cv2.waitKey(1)
            save_frame = False
            
            if key & 0xFF == ord('q') or key == 27:
                break
            elif key & 0xFF == ord(' '):
                recording = not recording
                print(f"Recording state: {recording}")
            elif key & 0xFF == ord('s'):
                save_frame = True 

            if recording or save_frame:
                frame_counter += 1
                
                # Only save if we've hit the frame skip interval or it's a manual save
                if save_frame or (frame_counter % args.frame_skip == 0):
                    frame_idx = frame_count
                
                    # Filenames
                    color_fname = f"frame_{frame_idx:05d}.jpg"
                    depth_fname = f"frame_{frame_idx:05d}.png" # Standard depth map
                    depth_npy_fname = f"frame_{frame_idx:05d}.npy" # POGS specialized raw depth

                    color_path = os.path.join(save_dirs['color'], color_fname)
                    depth_path = os.path.join(save_dirs['depth'], depth_fname)
                    depth_npy_path = os.path.join(save_dirs['depth_npy'], depth_npy_fname)

                    # Save Data
                    cv2.imwrite(color_path, color_image)
                    cv2.imwrite(depth_path, depth_image) # Save raw unit-16 depth png
                    np.save(depth_npy_path, depth_image.astype(np.float32) * depth_scale) # Save metric depth in meters

                    # Add to frames data for transforms.json
                    # Note: Without a tracker (like Vicon or SLAM), we perform a 'static' capture 
                    # or assume identity pose if just collecting data for later SfM processing (like colmap).
                    # POGS typically expects camera poses. 
                    # If we rely on Nerfstudio's built-in colmap, we just need images. 
                    # If we want to feed dummy poses for testing:
                    
                    frames_data.append({
                        "file_path": f"images/{color_fname}",
                        "depth_file_path": f"depth/{depth_fname}",
                        "transform_matrix": np.eye(4).tolist() # Identity pose placeholder
                    })

                    frame_count += 1
                    if save_frame:
                        print(f"Saved frame {frame_idx}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # Save transforms.json (Nerfstudio format)
        print("Saving transforms.json...")
        transforms_data = {
            "fl_x": intrinsics["fx"],
            "fl_y": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
            "w": intrinsics["width"],
            "h": intrinsics["height"],
            "camera_model": "OPENCV",
            "k1": intrinsics["distortion_params"][0],
            "k2": intrinsics["distortion_params"][1],
            "p1": intrinsics["distortion_params"][2],
            "p2": intrinsics["distortion_params"][3],
            "frames": frames_data
        }
        
        with open(os.path.join(scene_dir, "transforms.json"), "w") as f:
            json.dump(transforms_data, f, indent=4)
            
        print(f"Done! Captured {frame_count} frames to {scene_dir}")
        print(f"You can now process this data with Nerfstudio or POGS.")

if __name__ == "__main__":
    main()
