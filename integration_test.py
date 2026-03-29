"""
Integration test to verify proper shapes and data flow from POGS mock output
through the conversion bridge and into the ACTPCD model.
"""

import sys
import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add necessary paths to sys.path if running independently
sys.path.append(os.path.join(os.path.dirname(__file__), 'PointCloudMatters'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'pogs'))

from src.models.components.pcd_encoder.pointnet2 import PointNet2
from src.models.components.act.act import ACTPCD
from pogs_to_pointcloud import gaussians_to_pointcloud

class MockPOGSModel:
    """Mock POGS Model to generate fake Gaussians."""
    def __init__(self, num_points=2048, device="cpu"):
        self.device = torch.device(device)
        self.num_points = num_points
        self.opacities = torch.randn(num_points, 1, device=self.device)
        self.means = torch.randn(num_points, 3, device=self.device)
        self.features_dc = torch.randn(num_points, 3, device=self.device)
        self.gauss_params = {
            'dino_feats': torch.randn(num_points, 64, device=self.device)
        }
        self.cluster_labels = None
        self.mapping = None
        
        class MockConfig:
            sh_degree = 0
            gaussian_dim = 64
        self.config = MockConfig()
        
        self.nn = nn.Linear(64, 64).to(self.device)


def main():
    print("=== Testing POGS + PointNet++ + ACT Integration ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Output from mock POGS
    print("\n1. Generating fake POGS Gaussians...")
    pogs_model = MockPOGSModel(num_points=4096, device=device)
    
    # 2. Bridge
    print("2. Bridging Gaussians to PCM format...")
    # Expect 70 channels (6 for xyz+rgb, 64 for DINO)
    pcm_input_dict = gaussians_to_pointcloud(pogs_model, include_dino=True, opacity_threshold=0.01)
    
    print(f"  Coordinates shape (coord): {pcm_input_dict['coord'].shape}")
    print(f"  Features shape (feat): {pcm_input_dict['feat'].shape}")
    print(f"  Cumulative count (offset): {pcm_input_dict['offset'].shape}")
    
    # Check dimensions
    assert pcm_input_dict['feat'].shape[1] == 70, f"Expected 70 channels, got {pcm_input_dict['feat'].shape[1]}"
    
    # 3. Simulate ACT policy instantiation
    print("\n3. Instantiating ACTPCD model with PointNet++ backbone...")
    act = ACTPCD(
        backbone=PointNet2(in_channels=70, num_classes=0),
        transformer=nn.Identity(), # Replace with real transformer when doing actual weights
        encoder=None,
        state_dim=15, 
        num_queries=10, 
        action_dim=11,
        camera_names=["front"]
    ).to(device)
    
    print("4. Testing forward pass through backbone...")
    # Wrap in dict of dicts as ACT expects per-camera points
    pcds = {"front": pcm_input_dict}
    
    try:
        # Pcd_embeds handles passing through backbone and then downsampling (FPS/KNN)
        features, pos = act.forward_pcd_embed(pcds)
        print("  ✅ Backbone processed data successfully!")
        print(f"  Output features shape: {features.shape}")
        print(f"  Output positional encodings shape: {pos.shape}")
        print("\nAll integration interfaces verified!")
    except Exception as e:
        print(f"\n❌ Error during backbone integration pass:\n{e}")

if __name__ == "__main__":
    main()
