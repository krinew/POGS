import numpy as np
import cv2
import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Headless mock capture script (no GUI).")
    parser.add_argument("--scene_name", type=str, default="mock_test_headless", help="Name of the scene.")
    parser.add_argument("--save_path", type=str, default="data/realsense_captures", help="Base path to save data.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to generate")
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
    
    print(f"HEADLESS MODE: Generating {args.frames} frames...")
    print(f"Saving to: {os.path.join(args.save_path, args.scene_name)}")

    W, H = 1280, 720
    
    for i in range(args.frames):
        # 1. Simulate Moving Object (Circle)
        t = i * 0.1 
        
        # Color Image: Black background with moving green circle
        color_image = np.zeros((H, W, 3), dtype=np.uint8)
        center_x = int(W/2 + (W/4) * np.sin(t))
        center_y = int(H/2 + (H/4) * np.cos(t))
        cv2.circle(color_image, (center_x, center_y), 50, (0, 255, 0), -1)
        
        # 2. Simulate Depth Data
        # Background = 1 meter, Object = 0.5 meters
        depth_image_meters = np.ones((H, W), dtype=np.float32) * 1.0 
        
        # Create a mask for the circle area
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center < 50
        depth_image_meters[mask] = 0.5 
        
        # 3. Create Depth Visualization (PNG)
        # Normalize for visualization
        depth_normalized = (depth_image_meters / 2.0 * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # 4. Save to Disk
        img_path = os.path.join(save_dirs['img'], f"frame_{i:05d}.png")
        depth_npy_path = os.path.join(save_dirs['depth'], f"frame_{i:05d}.npy")
        depth_png_path = os.path.join(save_dirs['depth_png'], f"frame_{i:05d}.png")
        
        cv2.imwrite(img_path, color_image)
        np.save(depth_npy_path, depth_image_meters)
        cv2.imwrite(depth_png_path, depth_colormap)
        
        if i % 10 == 0:
            print(f"Saved frame {i}/{args.frames}")

    print(f"Done! Data saved to {os.path.join(args.save_path, args.scene_name)}")

if __name__ == "__main__":
    main()
