"""
Object extraction utilities for POGS - convert 3D Gaussians to point clouds.
"""

import numpy as np
import torch
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def extract_object_pointcloud_from_gaussians(
    pogs_model,
    cluster_label: int,
    min_opacity: float = 0.1,
    transform_to_metric: bool = True,
    dataset_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Extract point cloud for a single object from POGS 3D Gaussians.
    
    Args:
        pogs_model: POGSModel instance with trained Gaussians
        cluster_label: Target cluster label for object
        min_opacity: Minimum opacity threshold
        transform_to_metric: If True, scale to metric units
        dataset_scale: Scene scale for unit conversion
        
    Returns:
        (N, 3) point cloud in metric units, or None if empty
    """
    try:
        # Get cluster labels from POGS
        if pogs_model.cluster_labels is None:
            logger.warning("POGS clustering not available")
            return None
        
        cluster_labels = pogs_model.cluster_labels.cpu().numpy()
        
        # Get mask for this cluster
        cluster_mask = cluster_labels == cluster_label
        if not cluster_mask.any():
            logger.warning(f"Cluster {cluster_label} is empty")
            return None
        
        # Extract Gaussian parameters for this cluster
        means = pogs_model.means[cluster_mask].cpu().numpy()
        opacities = torch.sigmoid(pogs_model.opacities[cluster_mask]).cpu().numpy().squeeze()
        
        # Filter by opacity
        opacity_mask = opacities >= min_opacity
        if not opacity_mask.any():
            logger.warning(f"No Gaussians above opacity threshold {min_opacity}")
            return None
        
        points = means[opacity_mask]
        
        # Scale to metric
        if transform_to_metric:
            points = points / dataset_scale
        
        logger.info(f"Extracted {len(points)} points for cluster {cluster_label}")
        return points
        
    except Exception as e:
        logger.error(f"Failed to extract point cloud: {e}")
        return None


def extract_segmented_pointcloud(
    pogs_model,
    segmentation_2d: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    dataset_scale: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Extract point cloud using 2D segmentation mask and depth.
    
    Args:
        pogs_model: POGSModel instance
        segmentation_2d: (H, W) 2D segmentation mask (object=1, background=0)
        depth_image: (H, W) depth image in meters
        camera_intrinsics: (3, 3) camera K matrix
        camera_pose: (4, 4) camera-to-world transform
        dataset_scale: Scene scale for unit conversion
        
    Returns:
        (N, 3) point cloud in world frame (metric)
    """
    try:
        H, W = segmentation_2d.shape
        
        # Unproject depth to 3D
        y, x = np.where(segmentation_2d > 0)
        z = depth_image[y, x]
        
        # Filter invalid depths
        valid = z > 0
        x, y, z = x[valid], y[valid], z[valid]
        
        # Camera frame coordinates
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        X_cam = (x - cx) * z / fx
        Y_cam = (y - cy) * z / fy
        Z_cam = z
        
        points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)
        
        # Transform to world frame
        points_cam_h = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world_h = (camera_pose @ points_cam_h.T).T
        points_world = points_world_h[:, :3]
        
        # Scale to metric
        points_world = points_world / dataset_scale
        
        logger.info(f"Extracted {len(points_world)} points from segmentation")
        return points_world
        
    except Exception as e:
        logger.error(f"Failed to extract segmented point cloud: {e}")
        return None


def filter_pointcloud_by_bounds(
    points: np.ndarray,
    bounds: dict,
) -> np.ndarray:
    """
    Filter point cloud by bounding box.
    
    Args:
        points: (N, 3) point cloud
        bounds: Dict with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
        
    Returns:
        Filtered point cloud
    """
    mask = (
        (points[:, 0] >= bounds['x_min']) & (points[:, 0] <= bounds['x_max']) &
        (points[:, 1] >= bounds['y_min']) & (points[:, 1] <= bounds['y_max']) &
        (points[:, 2] >= bounds['z_min']) & (points[:, 2] <= bounds['z_max'])
    )
    return points[mask]


def downsample_pointcloud(
    points: np.ndarray,
    voxel_size: float = 0.01,
) -> np.ndarray:
    """
    Downsample point cloud using voxel grid.
    
    Args:
        points: (N, 3) point cloud
        voxel_size: Size of voxel grid
        
    Returns:
        Downsampled point cloud
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        downsampled = pcd.voxel_down_sample(voxel_size)
        return np.asarray(downsampled.points)
    except Exception as e:
        logger.warning(f"Failed to downsample with Open3D: {e}")
        return points


def compute_object_centroid(points: np.ndarray) -> np.ndarray:
    """
    Compute centroid of point cloud.
    
    Args:
        points: (N, 3) point cloud
        
    Returns:
        (3,) centroid coordinates
    """
    return np.mean(points, axis=0)


def align_pointcloud_to_origin(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate point cloud so centroid is at origin.
    
    Args:
        points: (N, 3) point cloud
        
    Returns:
        (aligned_points, centroid)
    """
    centroid = compute_object_centroid(points)
    aligned = points - centroid
    return aligned, centroid
