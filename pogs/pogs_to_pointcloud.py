import torch
import numpy as np

# Import SH2RGB exactly as POGS uses it
try:
    from nerfstudio.models.splatfacto import SH2RGB
except ImportError:
    # Fallback if not directly importable
    def SH2RGB(sh):
        return sh * 0.28209479177387814 + 0.5

def gaussians_to_pointcloud(
    model, 
    include_dino=True, 
    opacity_threshold=0.01,
    grid_size=0.005,
    obj_id=None
):
    """
    Converts a POGS Gaussian Splatting model into a PointCloudMatters-compatible input dict.
    This exactly replicates how POGS's _export_visible_gaussians exports to point clouds,
    while organizing the data into what ACTPCD's PointNet++ backbone expects.
    
    Args:
        model: A POGSModel instance containing the optimized gauss_params.
        include_dino: Whether to append DINO features.
        opacity_threshold: Filter out Gaussians with opacity below this threshold.
        grid_size: Voxel size for grid_coord calculation (matches PCM's GridSamplePCD).
        obj_id: Specific clustered object ID to extract (None for all).
        
    Returns:
        input_dict: Dictionary containing:
            - 'coord': (N, 3) point coordinates
            - 'feat': (N, C) point features (color normalized [-1, 1], plus optional DINO)
            - 'offset': (1,) cumulative point count (N,)
            - 'grid_coord': (N, 3) quantized voxel coordinates
    """
    with torch.no_grad():
        # 1. Filter by object ID or cluster mask if specified
        if obj_id is not None and model.cluster_labels is not None and model.mapping is not None:
            # POGS method of filtering down crop_ids
            crop_ids = torch.where(model.cluster_labels[model.keep_inds] == model.mapping[obj_id].item())[0]
        else:
            crop_ids = torch.arange(model.num_points, device=model.device)
            
        # 2. Extract base properties manually (avoiding rasterization overhead)
        opacities = model.opacities[crop_ids]
        means = model.means[crop_ids]
        features_dc = model.features_dc[crop_ids]
        dino_crop = model.gauss_params['dino_feats'][crop_ids] if 'dino_feats' in model.gauss_params else None
        
        # 3. Apply opacity filtering
        # POGS keeps all for point clouds, but we want to filter invisible fuzz for ACT
        # Convert sigmoid logic
        alphas = torch.sigmoid(opacities).squeeze(-1)
        valid_mask = alphas > opacity_threshold
        
        means = means[valid_mask]
        features_dc = features_dc[valid_mask]
        if dino_crop is not None:
            dino_crop = dino_crop[valid_mask]
            
        N = means.shape[0]
        if N == 0:
            # Handle empty cloud gracefully
            device = model.device
            return {
                "coord": torch.zeros((1, 3), device=device),
                "feat": torch.zeros((1, 6 + 64 if include_dino else 6), device=device),
                "grid_coord": torch.zeros((1, 3), dtype=torch.int32, device=device),
                "offset": torch.tensor([1], dtype=torch.int32, device=device)
            }
            
        # 4. Convert SH to RGB exactly as in _export_visible_gaussians
        if model.config.sh_degree > 0:
            colors = SH2RGB(features_dc)
        else:
            colors = torch.sigmoid(features_dc)
            
        # POGS pipeline normalizes colors min/max before writing to PLY
        # We match ACT's NormalizeColorPCD: color / 127.5 - 1
        # First, ensure it's in 0-255 range
        colors = torch.clamp(colors, 0.0, 1.0) * 255.0
        colors = colors / 127.5 - 1.0  # ACT's format [-1, 1]

        # 5. Build features array [color(3), coords(3)]
        # Based on ACT PCM pipeline's CollectPCD:
        feats_list = [colors, means]
        
        # 6. Include DINO
        if include_dino and dino_crop is not None:
            # DINO feats go through a small NN internally in get_outputs
            # Let's apply POGS's internal model.nn if needed for output feature matching
            # POGS: nn_inputs = dino_feats.view(-1, gaussian_dim) -> self.nn(nn_inputs)
            dino_nn_out = model.nn(dino_crop.view(-1, model.config.gaussian_dim))
            # Normalize DINO features just to be safe for PointNet++ input scale
            dino_norm = F.normalize(dino_nn_out, p=2, dim=-1)
            feats_list.append(dino_norm)
            
        feats = torch.cat(feats_list, dim=-1)
        
        # 7. Compute GridSamplePCD-like grid_coords
        scaled_coord = means / grid_size
        grid_coord = torch.floor(scaled_coord).to(torch.int32)
        grid_min = grid_coord.min(dim=0)[0]
        grid_coord = grid_coord - grid_min
        
        # 8. Assemble input dict
        input_dict = {
            "coord": means,                                 # (N, 3)
            "feat": feats,                                  # (N, 6) or (N, 70)
            "offset": torch.tensor([N], dtype=torch.int32, device=means.device), # batch size 1 natively
            "grid_coord": grid_coord                        # (N, 3)
        }
        
    return input_dict
