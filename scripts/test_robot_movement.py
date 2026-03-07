"""
Simple test script to verify OpenManipulator robot movement.

This script tests basic robot movements without the full POGS pipeline.
It sends simple joint commands and gripper commands to verify connectivity.

Usage:
    cd /home/pi0/POGS
    python scripts/test_robot_movement.py
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add POGS paths
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_with_lerobot():
    """Test using LeRobot OpenManipulator backend."""
    print("\n" + "="*60)
    print("Testing OpenManipulator with LeRobot Backend")
    print("="*60)
    
    try:
        from pogs.controller.open_manipulator import OpenManipulatorLeRobot
        
        # Initialize robot - try different gripper IDs
        print("\n[1] Connecting to robot on /dev/ttyUSB0...")
        
        # Try gripper ID 25 first (leader config), then ID 5 (follower config)
        gripper_ids_to_try = [25, 5]
        robot = None
        has_gripper = False
        
        for gripper_id in gripper_ids_to_try:
            try:
                print(f"  → Trying gripper ID {gripper_id}...")
                robot = OpenManipulatorLeRobot(
                    port="/dev/ttyUSB0",
                    use_leader_ids=True,
                    input_mode="normalized",
                    include_gripper=True,
                    gripper_id_override=gripper_id,
                )
                has_gripper = True
                print(f"  ✓ Connected with gripper ID {gripper_id}")
                break
            except RuntimeError as e:
                if "Missing motor IDs" in str(e):
                    print(f"  ✗ Gripper ID {gripper_id} not found")
                    continue
                else:
                    raise
        
        # If no gripper found, connect without it
        if robot is None:
            print("  ⚠ No gripper found, connecting without gripper...")
            robot = OpenManipulatorLeRobot(
                port="/dev/ttyUSB0",
                use_leader_ids=True,
                input_mode="normalized",
                include_gripper=False,
            )
            has_gripper = False
        
        print("✓ Robot connected successfully!")
        
        # Test 1: Read current position
        print("\n[2] Reading current joint positions...")
        try:
            positions = robot.bus.sync_read("Present_Position")
            print(f"✓ Current positions: {positions}")
        except Exception as e:
            print(f"⚠ Could not read positions: {e}")
        
        # Test 2: Open gripper
        print("\n[3] Opening gripper...")
        if has_gripper:
            robot.gripper.open()
            time.sleep(1.0)
            print("✓ Gripper opened")
        else:
            print("⚠ Skipped (no gripper)")
        
        # Test 3: Close gripper
        print("\n[4] Closing gripper...")
        if has_gripper:
            robot.gripper.close()
            time.sleep(1.0)
            print("✓ Gripper closed")
        else:
            print("⚠ Skipped (no gripper)")
        
        # Test 4: Move to home position (all zeros)
        print("\n[5] Moving to home position...")
        home_joints = [0.0, 0.0, 0.0, 0.0]  # [shoulder_pan, shoulder_lift, elbow, wrist]
        robot.move_joint(home_joints)
        time.sleep(2.0)
        print("✓ Moved to home position")
        
        # Test 5: Small movement test
        print("\n[6] Testing small joint movements...")
        
        # Move shoulder pan slightly
        test_positions = [
            ([10.0, 0.0, 0.0, 0.0], "Shoulder pan +10°"),
            ([-10.0, 0.0, 0.0, 0.0], "Shoulder pan -10°"),
            ([0.0, 10.0, 0.0, 0.0], "Shoulder lift +10"),
            ([0.0, 0.0, 0.0, 0.0], "Back to home"),
        ]
        
        for joints, desc in test_positions:
            print(f"  → {desc}: {joints}")
            robot.move_joint(joints)
            time.sleep(1.5)
        
        print("✓ Joint movement tests complete!")
        
        # Test 6: Open gripper at end
        print("\n[7] Opening gripper (final)...")
        if has_gripper:
            robot.gripper.open()
            time.sleep(0.5)
            print("✓ Gripper opened")
        else:
            print("⚠ Skipped (no gripper)")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - Robot is working!")
        print("="*60)
        
        return True
        
    except ImportError as e:
        print(f"✗ LeRobot import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Robot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_dynamixel_sdk():
    """Test using raw Dynamixel SDK - most reliable method."""
    print("\n" + "="*60)
    print("Testing OpenManipulator with Dynamixel SDK")
    print("="*60)
    
    try:
        import dynamixel_sdk as dxl
        
        DEVICENAME = '/dev/ttyUSB0'
        BAUDRATE = 1000000
        PROTOCOL_VERSION = 2.0
        
        # Control table addresses
        ADDR_TORQUE_ENABLE = 64
        ADDR_GOAL_POSITION = 116
        ADDR_PRESENT_POSITION = 132
        ADDR_PROFILE_VELOCITY = 112
        
        # Motor IDs (Leader arm)
        MOTOR_IDS = {
            'shoulder_pan': 21,
            'shoulder_lift': 22,
            'elbow_flex': 23,
            'wrist_flex': 24,
            'gripper': 25,
        }
        
        portHandler = dxl.PortHandler(DEVICENAME)
        packetHandler = dxl.PacketHandler(PROTOCOL_VERSION)
        
        print("\n[1] Connecting to robot on /dev/ttyUSB0...")
        if not portHandler.openPort():
            print("✗ Failed to open port")
            return False
        
        if not portHandler.setBaudRate(BAUDRATE):
            print("✗ Failed to set baudrate")
            return False
        
        print("✓ Port opened")
        
        # Scan for motors
        print("\n[2] Scanning for motors...")
        found = []
        for name, motor_id in MOTOR_IDS.items():
            model, comm, error = packetHandler.ping(portHandler, motor_id)
            if comm == dxl.COMM_SUCCESS:
                found.append(motor_id)
                print(f"  ✓ {name} (ID {motor_id}): Model {model}")
            else:
                print(f"  ✗ {name} (ID {motor_id}): Not found")
        
        if len(found) < 4:
            print("✗ Not enough motors found")
            portHandler.closePort()
            return False
        
        has_gripper = 25 in found
        
        # Enable torque for all found motors
        print("\n[3] Enabling torque...")
        for motor_id in found:
            packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 1)
        print("✓ Torque enabled")
        
        # Helper function to move a motor
        def move_motor(motor_id, position, velocity=100):
            # Set velocity
            packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_PROFILE_VELOCITY, velocity)
            # Set goal position
            packetHandler.write4ByteTxRx(portHandler, motor_id, ADDR_GOAL_POSITION, position)
        
        def read_position(motor_id):
            pos, _, _ = packetHandler.read4ByteTxRx(portHandler, motor_id, ADDR_PRESENT_POSITION)
            return pos
        
        # Position values (2048 = center/zero position for XM430)
        CENTER = 2048
        
        # Test gripper
        if has_gripper:
            print("\n[4] Testing gripper...")
            print("  → Opening gripper...")
            move_motor(25, CENTER + 500, velocity=200)  # Open
            time.sleep(1.0)
            print("  → Closing gripper...")
            move_motor(25, CENTER - 300, velocity=200)  # Close
            time.sleep(1.0)
            print("  → Opening gripper (final)...")
            move_motor(25, CENTER + 500, velocity=200)  # Open
            time.sleep(0.5)
            print("✓ Gripper test complete")
        else:
            print("\n[4] Gripper not found, skipping...")
        
        # Test arm movement
        print("\n[5] Testing arm movement...")
        
        # Read current positions
        positions = {name: read_position(mid) for name, mid in MOTOR_IDS.items() if mid in found}
        print(f"  Current positions: {positions}")
        
        # Move shoulder pan
        print("  → Moving shoulder pan +10°...")
        move_motor(21, CENTER + 114, velocity=100)  # ~10 degrees (4096/360 ≈ 11.4 counts per degree)
        time.sleep(1.5)
        
        print("  → Moving shoulder pan -10°...")
        move_motor(21, CENTER - 114, velocity=100)
        time.sleep(1.5)
        
        print("  → Moving back to center...")
        move_motor(21, CENTER, velocity=100)
        time.sleep(1.5)
        
        print("✓ Arm movement test complete")
        
        # Disable torque
        print("\n[6] Disabling torque...")
        for motor_id in found:
            packetHandler.write1ByteTxRx(portHandler, motor_id, ADDR_TORQUE_ENABLE, 0)
        
        portHandler.closePort()
        
        print("\n" + "="*60)
        print("✓ ALL SDK TESTS PASSED - Robot is working!")
        print("="*60)
        
        return True
        
    except ImportError as e:
        print(f"✗ Dynamixel SDK import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Robot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_grasp_execution():
    """Test grasp execution with mock grasps (no actual grasp generation)."""
    print("\n" + "="*60)
    print("Testing Mock Grasp Execution")
    print("="*60)
    
    try:
        from pogs.controller.open_manipulator import OpenManipulatorLeRobot
        from scipy.spatial.transform import Rotation as R
        
        # Initialize robot - try with gripper, fallback to without
        print("\n[1] Connecting to robot...")
        
        gripper_ids_to_try = [25, 5]
        robot = None
        has_gripper = False
        
        for gripper_id in gripper_ids_to_try:
            try:
                print(f"  → Trying gripper ID {gripper_id}...")
                robot = OpenManipulatorLeRobot(
                    port="/dev/ttyUSB0",
                    use_leader_ids=True,
                    input_mode="normalized",
                    include_gripper=True,
                    gripper_id_override=gripper_id,
                )
                has_gripper = True
                print(f"  ✓ Connected with gripper ID {gripper_id}")
                break
            except RuntimeError as e:
                if "Missing motor IDs" in str(e):
                    print(f"  ✗ Gripper ID {gripper_id} not found")
                    continue
                else:
                    raise
        
        if robot is None:
            print("  ⚠ No gripper found, connecting without gripper...")
            robot = OpenManipulatorLeRobot(
                port="/dev/ttyUSB0",
                use_leader_ids=True,
                input_mode="normalized",
                include_gripper=False,
            )
        
        print("✓ Robot connected!")
        
        # Generate a simple mock grasp pose
        print("\n[2] Generating mock grasp pose...")
        
        # A simple grasp pose in front of the robot
        grasp_pose = np.eye(4)
        grasp_pose[:3, 3] = [0.2, 0.0, 0.1]  # x=20cm, y=0, z=10cm
        grasp_pose[:3, :3] = R.from_euler('xyz', [0, -1.57, 0]).as_matrix()  # Point down
        
        print(f"✓ Mock grasp pose:\n{grasp_pose}")
        
        # For 4-DOF robot, we need to compute IK
        # For now, just do simple movements
        print("\n[3] Executing simplified grasp sequence...")
        
        # Pre-grasp: home position with gripper open
        print("  → Pre-grasp: Opening gripper...")
        robot.gripper.open()
        time.sleep(1.0)
        
        # Move to approach position
        print("  → Approach: Moving to position...")
        robot.move_joint([0.0, 20.0, -20.0, 0.0])  # Simple reach forward
        time.sleep(2.0)
        
        # Close gripper (grasp)
        print("  → Grasp: Closing gripper...")
        robot.gripper.close()
        time.sleep(1.0)
        
        # Retract
        print("  → Retract: Moving up...")
        robot.move_joint([0.0, 0.0, 0.0, 0.0])  # Back to home
        time.sleep(2.0)
        
        # Release
        print("  → Release: Opening gripper...")
        robot.gripper.open()
        time.sleep(1.0)
        
        print("\n" + "="*60)
        print("✓ Mock grasp sequence complete!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Mock grasp test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*60)
    print("#" + " "*20 + "ROBOT MOVEMENT TEST" + " "*19 + "#")
    print("#"*60)
    
    # Check port access
    import os
    port = "/dev/ttyUSB0"
    if os.path.exists(port):
        print(f"\n✓ Port {port} exists")
        if os.access(port, os.R_OK | os.W_OK):
            print(f"✓ Port {port} is accessible (read/write)")
        else:
            print(f"⚠ Port {port} exists but may not be accessible")
            print(f"  Try: sudo chmod 666 {port}")
    else:
        print(f"\n✗ Port {port} not found!")
        print("  Make sure the robot is connected via USB")
        return
    
    # Menu
    print("\n" + "-"*60)
    print("Select test mode:")
    print("  1. Basic movement test (LeRobot backend)")
    print("  2. Basic movement test (Dynamixel SDK)")
    print("  3. Mock grasp execution")
    print("  4. Run all tests")
    print("-"*60)
    
    try:
        choice = input("\nEnter choice (1-4) [default=1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    
    if choice == "1":
        test_with_lerobot()
    elif choice == "2":
        test_with_dynamixel_sdk()
    elif choice == "3":
        test_mock_grasp_execution()
    elif choice == "4":
        print("\nRunning all tests...")
        test_with_lerobot()
        test_mock_grasp_execution()
    else:
        print("Invalid choice. Running default test...")
        test_with_lerobot()


if __name__ == "__main__":
    main()
