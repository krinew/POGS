import pickle
from pathlib import Path
import cv2
import numpy as np

ep_path = "/home/pi0/POGS-ACT-implementation/POGS/PointCloudMatters/data/rlbench/raw/train/open_drawer/all_variations/episodes/episode0/low_dim_obs.pkl"
import rlbench.backend.observation as observation
with open(ep_path, 'rb') as f:
    obs = pickle.load(f)

# The observations are actually saved in other files or inside low_dim_obs?
# Wait, RLBench saves observations individually or as a demo?
print(len(obs))
