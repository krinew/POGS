import numpy as np
import cv2
import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Mock capture script to test workflow without RealSense camera.")
    parser.add_argument("--scene_name", type=str, default="mock_test", help="Name of the scene.")
    parser.add_argument("--save_path", type=str, default="data/realsense_captures", help="Base path to save data.")
    return parser.parse_args()

def setup_directories(base_path, scene_name):
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
    
    print(f"MOCK MODE: Saving data to: {os.path.join(args.save_path, args.scene_name)}")
    print("Simulating Intel RealSense stream...")

    W, H = 1280, 720
    depth_scale = 0.001 # Standard 1mm scale

    print("\n--- Controls ---")
    print("Press [Space] to toggle recording ON/OFF.")
    print("Press [s] to save a single frame.")
    print("Press [q] or [ESC] to quit.")
    
    frame_count = 0
    recording = False
    
    try:
        while True:
            # Generate fake data based on time (moving circle)
            t = time.time() * 2
            
            # Moving circle pattern coordinates
            center_x = int(W/2 + (W/4) * np.sin(t))
            center_y = int(H/2 + (H/4) * np.cos(t))
            
            # 1. Color Image: Dark background with moving green ball
            color_image = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)
            cv2.putText(color_image, "MOCK DATA - TESTING", (center_x-80, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            # 2. Depth Image: 
            # Background = 2 meters (2000mm)
            # Ball = 1 meter (1000mm)
            depth_image_raw = np.ones((H, W), dtype=np.uint16) * 2000 
            cv2.circle(depth_image_raw, (center_x, center_y), 50, 1000, -1)

            # Convert depth to meters (float32) used by POGS
            depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale

            # Visualization for the window
            depth_image_viz = cv2.normalize(depth_image_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colormap = cv2.applyColorMap(depth_image_viz, cv2.COLORMAP_JET)

            # Stack images side-by-side
            images = np.hstack((color_image, depth_colormap))

            # UI Text
            status_color = (0, 255, 0) if recording else (0, 0, 255)
            cv2.putText(images, f"Recording: {recording} | Frames: {frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            cv2.imshow('MOCK RealSense (Press SPACE to record)', images)
            
            # Wait 33ms (~30 FPS)
            key = cv2.waitKey(33) 
            
            save_frame = False
            if key & 0xFF == ord('q') or key == 27:
                break
            elif key & 0xFF == ord(' '):
                recording = not recording
                print(f"Recording state: {recording}")
            elif key & 0xFF == ord('s'):
                save_frame = True
            
            if recording or save_frame:
                frame_idx = frame_count
                frame_count += 1
                
                img_path = os.path.join(save_dirs['img'], f"frame_{frame_idx:05d}.png")
                depth_npy_path = os.path.join(save_dirs['depth'], f"frame_{frame_idx:05d}.npy")
                depth_png_path = os.path.join(save_dirs['depth_png'], f"frame_{frame_idx:05d}.png")
                
                # Save data exactly how the real script does
                cv2.imwrite(img_path, color_image)
                np.save(depth_npy_path, depth_image_meters)
                cv2.imwrite(depth_png_path, depth_colormap)
                
                if save_frame:
                    print(f"Saved mock frame {frame_idx}")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print(f"Test complete. Saved {frame_count} frames to {os.path.join(args.save_path, args.scene_name)}")

if __name__ == "__main__":
    main()
