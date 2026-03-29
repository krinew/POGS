import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCloudDecoder(nn.Module):
    """
    FoldingNet-style decoder to reconstruct a point cloud from a global embedding.
    Useful for evaluating whether the PointNet++ embedding preserves spatial structure.
    
    Paper: "FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation"
    """
    
    def __init__(self, embedding_dim=1024, num_points=2048):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_points = num_points
        
        # We start with a 2D grid to "fold" into 3D
        # Number of grid points = sqrt(num_points) x sqrt(num_points)
        grid_size = int(torch.ceil(torch.sqrt(torch.tensor(float(num_points))))).item()
        x = torch.linspace(-1, 1, grid_size)
        y = torch.linspace(-1, 1, grid_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        
        # Flatten and truncate to exact num_points
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        grid = grid[:num_points, :]  # (num_points, 2)
        
        # Register as buffer so it moves to correct device
        self.register_buffer('grid', grid)
        
        # Folding Layer 1: (grid_2d + embedding) -> 3D intermediate
        self.fold1 = nn.Sequential(
            nn.Linear(2 + embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
        # Folding Layer 2: (intermediate_3d + embedding) -> 3D output
        self.fold2 = nn.Sequential(
            nn.Linear(3 + embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
    def forward(self, embedding):
        """
        Args:
            embedding: (B, embedding_dim) Global feature vector
            
        Returns:
            reconstructed: (B, num_points, 3) Reconstructed point cloud
        """
        B = embedding.size(0)
        
        # Repeat embedding for each point in the grid
        # (B, 1, embedding_dim) -> (B, num_points, embedding_dim)
        emb_rep = embedding.unsqueeze(1).repeat(1, self.num_points, 1)
        
        # Repeat grid for each element in the batch
        # (num_points, 2) -> (B, num_points, 2)
        grid_rep = self.grid.unsqueeze(0).repeat(B, 1, 1)
        
        # --- First Fold ---
        # Concat grid with embedding: (B, num_points, 2 + embedding_dim)
        feat1 = torch.cat([grid_rep, emb_rep], dim=-1)
        
        # We need to flatten to (B * num_points, C) for BatchNorm1d to work correctly
        feat1_flat = feat1.view(-1, 2 + self.embedding_dim)
        out1_flat = self.fold1(feat1_flat)
        out1 = out1_flat.view(B, self.num_points, 3)
        
        # --- Second Fold ---
        # Concat intermediate 3D with embedding: (B, num_points, 3 + embedding_dim)
        feat2 = torch.cat([out1, emb_rep], dim=-1)
        
        feat2_flat = feat2.view(-1, 3 + self.embedding_dim)
        out2_flat = self.fold2(feat2_flat)
        out2 = out2_flat.view(B, self.num_points, 3)
        
        return out2


def chamfer_distance(p1, p2):
    """
    Computes the Chamfer Distance between two point clouds.
    
    Args:
        p1: (B, N, 3)
        p2: (B, M, 3)
        
    Returns:
        dist: (B,) Scalar chamfer distance per batch element
    """
    # (B, N, 1, 3) - (B, 1, M, 3) -> (B, N, M, 3) -> (B, N, M)
    diff = p1.unsqueeze(2) - p2.unsqueeze(1)
    dist_matrix = torch.sum(diff**2, dim=-1)
    
    # Min distance from each point in p1 to any point in p2
    min_dist_p1_to_p2 = torch.min(dist_matrix, dim=2)[0]  # (B, N)
    
    # Min distance from each point in p2 to any point in p1
    min_dist_p2_to_p1 = torch.min(dist_matrix, dim=1)[0]  # (B, M)
    
    # Mean over points, then sum both directions
    cd = torch.mean(min_dist_p1_to_p2, dim=1) + torch.mean(min_dist_p2_to_p1, dim=1)
    
    return cd
