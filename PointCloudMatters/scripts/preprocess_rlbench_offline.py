import os
import pickle
import glob
import numpy as np
import cv2
from tqdm import tqdm
import torch
import clip

root = "data/rlbench/raw"
save_root = "data/rlbench/processed"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = "ViT-B/16"
clip_model, _ = clip.load(
    clip_model, device=device, download_root=os.path.expanduser("~/.cache/clip")
)
clip_model.requires_grad_(False)
clip_model.eval()

task_names = ["open_drawer"]
camera_views = ["front"]
modalities = ["rgb", "depth", "mask"]
low_dim_states = ["joint_velocities", "joint_positions", "joint_forces", "task_low_dim_state"]
gripper_states = ["gripper_open", "gripper_pose", "gripper_matrix", "gripper_joint_positions", "gripper_touch_forces"]

def image_to_float_array(img: np.ndarray, float_depth: float) -> np.ndarray:
    """RLBench internal depth decoding."""
    img = np.array(img, dtype=np.float32)
    depth = img[..., 0] + img[..., 1] * 256.0 + img[..., 2] * 256.0**2
    return depth / (256.0**3 - 1) * float_depth

for stage in ["train", "val"]:
    for task_name in task_names:
        print(f"\nProcessing {stage} data of task {task_name}...")
        
        task_dir = os.path.join(root, stage, task_name, "variation-1", "episodes")
        if not os.path.exists(task_dir):
            print(f"Directory not found: {task_dir}")
            continue
            
        episodes = glob.glob(os.path.join(task_dir, "episode*"))
        os.makedirs(os.path.join(save_root, stage, task_name), exist_ok=True)
        
        for ep_path in tqdm(episodes):
            ep_num = int(os.path.basename(ep_path).replace("episode", ""))
            
            low_dim_file = os.path.join(ep_path, "low_dim_obs.pkl")
            desc_file = os.path.join(ep_path, "variation_descriptions.pkl")
            
            if not os.path.exists(low_dim_file):
                continue
                
            demo_obs = pickle.load(open(low_dim_file, "rb"))
            descriptions = pickle.load(open(desc_file, "rb"))
            
            description_token = clip.tokenize(descriptions[0]).to(device)
            task_goal = clip_model.encode_text(description_token).cpu().numpy()[0]
            
            demo_array = []
            for i, frame in enumerate(demo_obs):
                frame_dict = {"ignore_collisions": frame.ignore_collisions}
                
                # Fetch cameras explicitly from the saved images
                for view in camera_views:
                    # RGB
                    img_path = os.path.join(ep_path, f"{view}_rgb", f"{i}.png")
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        frame_dict[f"{view}_rgb"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                    # Depth
                    depth_path = os.path.join(ep_path, f"{view}_depth", f"{i}.png")
                    if os.path.exists(depth_path):
                        dimg = cv2.imread(depth_path)
                        # The first frame of the demo has camera far bounds in misc dict
                        far = demo_obs[0].misc[f"{view}_camera_far"]
                        near = demo_obs[0].misc[f"{view}_camera_near"]
                        d_array = image_to_float_array(dimg, far)
                        frame_dict[f"{view}_depth"] = d_array
                        
                for state in low_dim_states + gripper_states:
                    if hasattr(frame, state):
                        frame_dict[state] = getattr(frame, state)
                        
                demo_array.append(frame_dict)
                
            save_path = os.path.join(save_root, stage, task_name, f"ep{ep_num}.npy")
            np.save(save_path, dict(demo=demo_array, task_goal=task_goal), allow_pickle=True)
            
print("\nDone processing offline!")
