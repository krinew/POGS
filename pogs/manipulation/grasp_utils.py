"""
Grasp filtering and trajectory planning utilities for 4-DOF pick-and-place.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def project_6dof_to_4dof_grasps(
    pred_grasps_6dof: np.ndarray,
    fixed_pitch: float = 1.57,
) -> np.ndarray:
    """
    Project 6-DOF grasps to 4-DOF by constraining pitch and roll.
    
    Args:
        pred_grasps_6dof: (N, 4, 4) array of 6-DOF grasp poses
        fixed_pitch: Fixed pitch angle (default: pi/2 for vertical)
        
    Returns:
        (N, 4, 4) array of 4-DOF projected grasps
    """
    from pogs.controller.robot_interface import project_pose_to_4dof
    
    proj_grasps = []
    for grasp in pred_grasps_6dof:
        try:
            proj = project_pose_to_4dof(grasp, fixed_pitch=fixed_pitch)
            proj_grasps.append(proj)
        except Exception as e:
            logger.warning(f"Failed to project grasp: {e}")
            continue
    
    return np.array(proj_grasps) if proj_grasps else np.array([])


def check_grasp_collision(
    grasp_pose: np.ndarray,
    table_height: float = 0.0,
    gripper_offset: float = 0.1,
) -> bool:
    """
    Check if grasp pose collides with table or workspace boundaries.
    
    Args:
        grasp_pose: (4, 4) grasp pose matrix
        table_height: Height of table in world frame
        gripper_offset: Distance from TCP to gripper fingers
        
    Returns:
        True if collision-free, False otherwise
    """
    tcp_z = grasp_pose[2, 3]
    
    # Check table collision
    if tcp_z < table_height + gripper_offset:
        logger.debug(f"Grasp below table height: {tcp_z:.3f} < {table_height + gripper_offset:.3f}")
        return False
    
    # TODO: Add more sophisticated collision checking
    # - Check workspace bounds
    # - Check self-collision for robot arm
    # - Use MoveIt or similar planning library
    
    return True


def filter_grasps_by_feasibility(
    pred_grasps_6dof: np.ndarray,
    scores: np.ndarray,
    score_threshold: float = 0.3,
    table_height: float = 0.0,
    max_grasps: int = 10,
    fixed_pitch: float = 1.57,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter grasps for 4-DOF feasibility and return ranked list.
    
    Args:
        pred_grasps_6dof: (N, 4, 4) 6-DOF grasp poses
        scores: (N,) grasp scores
        score_threshold: Minimum score to consider
        table_height: Table height for collision checking
        max_grasps: Maximum grasps to return
        fixed_pitch: Pitch constraint for 4-DOF projection
        
    Returns:
        (feasible_grasps_4dof, feasible_scores, feasible_indices)
    """
    logger.info(f"Filtering {len(pred_grasps_6dof)} grasps...")
    
    # Filter by score threshold
    valid_mask = scores >= score_threshold
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        logger.warning(f"No grasps above score threshold {score_threshold}")
        return np.array([]), np.array([]), np.array([])
    
    valid_grasps = pred_grasps_6dof[valid_indices]
    valid_scores = scores[valid_indices]
    
    # Project to 4-DOF
    proj_grasps = project_6dof_to_4dof_grasps(valid_grasps, fixed_pitch=fixed_pitch)
    
    if len(proj_grasps) == 0:
        logger.warning("Failed to project any grasps to 4-DOF")
        return np.array([]), np.array([]), np.array([])
    
    # Filter by collision
    feasible_mask = np.array([
        check_grasp_collision(g, table_height=table_height) 
        for g in proj_grasps
    ])
    feasible_indices_local = np.where(feasible_mask)[0]
    
    if len(feasible_indices_local) == 0:
        logger.warning("No collision-free grasps found")
        return np.array([]), np.array([]), np.array([])
    
    feasible_grasps = proj_grasps[feasible_indices_local]
    feasible_scores = valid_scores[feasible_indices_local]
    feasible_indices_global = valid_indices[feasible_indices_local]
    
    # Sort by score (descending)
    sort_idx = np.argsort(-feasible_scores)[:max_grasps]
    
    logger.info(f"Filtered to {len(sort_idx)} feasible grasps")
    
    return (
        feasible_grasps[sort_idx],
        feasible_scores[sort_idx],
        feasible_indices_global[sort_idx]
    )


def select_best_grasp(
    feasible_grasps_4dof: np.ndarray,
    feasible_scores: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Select the grasp with highest score.
    
    Args:
        feasible_grasps_4dof: (N, 4, 4) feasible 4-DOF grasps
        feasible_scores: (N,) scores for each grasp
        
    Returns:
        (best_grasp, best_score)
    """
    if len(feasible_grasps_4dof) == 0:
        return None, 0.0
    
    best_idx = 0
    best_grasp = feasible_grasps_4dof[best_idx]
    best_score = feasible_scores[best_idx]
    
    logger.info(f"Selected best grasp with score {best_score:.3f}")
    return best_grasp, best_score


def plan_linear_trajectory(
    start_pose: np.ndarray,
    end_pose: np.ndarray,
    num_waypoints: int = 10,
    approach_distance: float = 0.05,
) -> List[np.ndarray]:
    """
    Plan linear trajectory with approach phase.
    
    Args:
        start_pose: (4, 4) starting pose
        end_pose: (4, 4) target pose
        num_waypoints: Number of interpolation steps
        approach_distance: Distance to approach before final pose
        
    Returns:
        List of (4, 4) waypoint poses
    """
    waypoints = []
    
    # Approach waypoint
    approach_pose = end_pose.copy()
    approach_pose[2, 3] += approach_distance
    
    # Interpolate from start to approach
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        interp_pose = start_pose.copy()
        interp_pose[:3, 3] = (1 - t) * start_pose[:3, 3] + t * approach_pose[:3, 3]
        # TODO: Interpolate rotation smoothly (SLERP)
        waypoints.append(interp_pose)
    
    # Final grasp pose
    waypoints.append(end_pose)
    
    return waypoints


def compute_place_location(
    pick_pose: np.ndarray,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute place location based on pick location.
    
    Args:
        pick_pose: (4, 4) original pick pose
        offset: (3,) offset from pick location (default: [0.2, 0, 0] for 20cm offset)
        
    Returns:
        (4, 4) place pose
    """
    if offset is None:
        offset = np.array([0.2, 0.0, 0.0])
    
    place_pose = pick_pose.copy()
    place_pose[:3, 3] += offset
    
    return place_pose
