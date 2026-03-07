"""
Test: GraspNet → Robot Execution Pipeline using LeRobot OmxFollower

This script tests the full grasp pipeline:
1. Generate mock grasps (simulating GraspNet output)
2. Use LeRobot OmxFollower for robot control
3. Execute pick-and-place sequence

Note: Since OMX doesn't have URDF/IK in LeRobot yet, we use joint-space control
with predefined grasp positions. For full IK, add OMX URDF to LeRobot.

Usage:
    cd /home/pi0/POGS
    python scripts/test_graspnet_to_robot.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "BTP_OMX_Lerobot/lerobot/src"))


def generate_mock_grasps(n_grasps: int = 5) -> list:
    """
    Generate mock grasps simulating GraspNet output.
    
    Returns joint-space grasps for OMX (since we don't have IK).
    Format: [shoulder_pan_deg, shoulder_lift_norm, elbow_flex_norm, wrist_flex_norm]
    """
    print(f"\n[GraspNet] Generating {n_grasps} mock grasps (joint-space)...")
    
    grasps = []
    for i in range(n_grasps):
        # Random grasp in joint space (normalized values)
        shoulder_pan = np.random.uniform(-30, 30)      # degrees
        shoulder_lift = np.random.uniform(-20, 20)     # normalized [-100, 100]
        elbow_flex = np.random.uniform(-30, 10)        # normalized 
        wrist_flex = np.random.uniform(-20, 20)        # normalized
        
        score = np.random.uniform(0.6, 1.0)
        
        grasp = {
            'joints': {
                'shoulder_pan': shoulder_pan,
                'shoulder_lift': shoulder_lift,
                'elbow_flex': elbow_flex,
                'wrist_flex': wrist_flex,
            },
            'score': score,
        }
        grasps.append(grasp)
        
        print(f"  Grasp {i+1}: pan={shoulder_pan:.1f}°, lift={shoulder_lift:.1f}, "
              f"elbow={elbow_flex:.1f}, wrist={wrist_flex:.1f}, score={score:.3f}")
    
    # Sort by score
    grasps.sort(key=lambda x: x['score'], reverse=True)
    return grasps


def test_with_omx_follower():
    """Test using LeRobot OmxFollower class."""
    print("\n" + "="*60)
    print("Testing GraspNet → Robot with LeRobot OmxFollower")
    print("="*60)
    
    try:
        from lerobot.robots.omx_follower import OmxFollower, Omx_FollowerConfig
        
        # Configure OMX follower
        print("\n[1] Connecting to robot via LeRobot OmxFollower...")
        config = Omx_FollowerConfig(
            port="/dev/ttyUSB0",
            id="omx",
            disable_torque_on_disconnect=True,
        )
        
        robot = OmxFollower(config)
        robot.connect(calibrate=False)
        print("✓ Robot connected!")
        
        # Generate mock grasps
        grasps = generate_mock_grasps(n_grasps=3)
        
        # Execute best grasp
        best_grasp = grasps[0]
        print(f"\n[2] Executing best grasp (score={best_grasp['score']:.3f})...")
        
        # Home position
        home_action = {
            'shoulder_pan.pos': 0.0,
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 0.0,
            'gripper.pos': 100.0,  # Open
        }
        
        # Pre-grasp position (approach)
        pre_grasp_action = {
            'shoulder_pan.pos': best_grasp['joints']['shoulder_pan'],
            'shoulder_lift.pos': best_grasp['joints']['shoulder_lift'] - 10,  # Slightly higher
            'elbow_flex.pos': best_grasp['joints']['elbow_flex'] + 10,
            'wrist_flex.pos': best_grasp['joints']['wrist_flex'],
            'gripper.pos': 100.0,  # Open
        }
        
        # Grasp position
        grasp_action = {
            'shoulder_pan.pos': best_grasp['joints']['shoulder_pan'],
            'shoulder_lift.pos': best_grasp['joints']['shoulder_lift'],
            'elbow_flex.pos': best_grasp['joints']['elbow_flex'],
            'wrist_flex.pos': best_grasp['joints']['wrist_flex'],
            'gripper.pos': 100.0,  # Still open
        }
        
        # Execute sequence
        print("\n[3] Executing grasp sequence...")
        
        print("  → Moving to home...")
        robot.send_action(home_action)
        time.sleep(2.0)
        
        print("  → Opening gripper...")
        robot.send_action({**home_action, 'gripper.pos': 100.0})
        time.sleep(1.0)
        
        print("  → Moving to pre-grasp...")
        robot.send_action(pre_grasp_action)
        time.sleep(2.0)
        
        print("  → Moving to grasp position...")
        robot.send_action(grasp_action)
        time.sleep(2.0)
        
        print("  → Closing gripper (grasping)...")
        robot.send_action({**grasp_action, 'gripper.pos': 0.0})
        time.sleep(1.5)
        
        print("  → Retracting...")
        robot.send_action({**pre_grasp_action, 'gripper.pos': 0.0})
        time.sleep(2.0)
        
        # Place at opposite side
        place_action = {
            'shoulder_pan.pos': -best_grasp['joints']['shoulder_pan'],
            'shoulder_lift.pos': 0.0,
            'elbow_flex.pos': 0.0,
            'wrist_flex.pos': 0.0,
            'gripper.pos': 0.0,
        }
        
        print("  → Moving to place position...")
        robot.send_action(place_action)
        time.sleep(2.0)
        
        print("  → Opening gripper (releasing)...")
        robot.send_action({**place_action, 'gripper.pos': 100.0})
        time.sleep(1.0)
        
        print("  → Returning home...")
        robot.send_action(home_action)
        time.sleep(2.0)
        
        # Disconnect
        robot.disconnect()
        
        print("\n" + "="*60)
        print("✓ GRASPNET → ROBOT PIPELINE COMPLETE!")
        print("="*60)
        
        return True
        
    except ImportError as e:
        print(f"✗ LeRobot import error: {e}")
        print("  Falling back to direct Dynamixel control...")
        return test_with_direct_dynamixel()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_direct_dynamixel():
    """Fallback: Test using direct Dynamixel SDK control."""
    print("\n" + "="*60)
    print("Testing GraspNet → Robot with Direct Dynamixel SDK")
    print("="*60)
    
    try:
        from pogs.controller.open_manipulator import OpenManipulatorLeRobot
        
        # Connect (use leader IDs since motors are IDs 21-25)
        print("\n[1] Connecting to robot...")
        robot = OpenManipulatorLeRobot(
            port="/dev/ttyUSB0",
            use_leader_ids=True,  # Leader IDs: 21-25 (your robot config)
            input_mode="normalized",
            include_gripper=True,
        )
        print("✓ Robot connected!")
        
        # Generate mock grasps
        grasps = generate_mock_grasps(n_grasps=3)
        best = grasps[0]
        
        print(f"\n[2] Executing best grasp (score={best['score']:.3f})...")
        
        # Home
        print("  → Moving to home...")
        robot.move_joint([0.0, 0.0, 0.0, 0.0])
        robot.gripper.open()
        time.sleep(2.0)
        
        # Pre-grasp
        print("  → Moving to pre-grasp...")
        robot.move_joint([
            best['joints']['shoulder_pan'],
            best['joints']['shoulder_lift'] - 10,
            best['joints']['elbow_flex'] + 10,
            best['joints']['wrist_flex'],
        ])
        time.sleep(2.0)
        
        # Grasp
        print("  → Moving to grasp position...")
        robot.move_joint([
            best['joints']['shoulder_pan'],
            best['joints']['shoulder_lift'],
            best['joints']['elbow_flex'],
            best['joints']['wrist_flex'],
        ])
        time.sleep(2.0)
        
        print("  → Closing gripper...")
        robot.gripper.close()
        time.sleep(1.5)
        
        # Retract
        print("  → Retracting...")
        robot.move_joint([
            best['joints']['shoulder_pan'],
            best['joints']['shoulder_lift'] - 10,
            best['joints']['elbow_flex'] + 10,
            best['joints']['wrist_flex'],
        ])
        time.sleep(2.0)
        
        # Place
        print("  → Moving to place position...")
        robot.move_joint([
            -best['joints']['shoulder_pan'],
            0.0, 0.0, 0.0,
        ])
        time.sleep(2.0)
        
        print("  → Opening gripper...")
        robot.gripper.open()
        time.sleep(1.0)
        
        # Home
        print("  → Returning home...")
        robot.move_joint([0.0, 0.0, 0.0, 0.0])
        time.sleep(2.0)
        
        print("\n" + "="*60)
        print("✓ GRASPNET → ROBOT PIPELINE COMPLETE!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*60)
    print("#" + " "*12 + "GRASPNET → ROBOT TEST" + " "*13 + "#")
    print("#"*60)
    
    # Check port
    import os
    port = "/dev/ttyUSB0"
    if not os.path.exists(port):
        print(f"\n✗ Robot port {port} not found!")
        print("  Make sure the robot is connected via USB.")
        return
    
    print(f"\n✓ Robot port {port} found")
    
    # Use direct Dynamixel control (works with leader IDs 21-25)
    # Note: OmxFollower uses follower IDs 1-5, but your robot has leader IDs
    test_with_direct_dynamixel()


if __name__ == "__main__":
    main()
