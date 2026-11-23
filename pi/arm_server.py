# main.py
"""
Main Application Script
----------------------------
Example code for the MP1 RRMC implementation
"""

import time
import socket
import struct
import traceback
import numpy as np

from funrobo_hiwonder.core.hiwonder import HiwonderRobot
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.fivedof import FiveDOF
#from funrobo_hiwonder.examples.joystick_control import joystick_control

def main():
    """ Main loop that reads UDP target positions and moves the robot to them. """
    # UDP: laptop sends '<3f' = target [x, y, z] (12 bytes) to port 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5005))
    try:

        # Initialize components
        robot = HiwonderRobot()
        model = FiveDOF()
        
        control_hz = 20
        dt = 1 / control_hz
        t0 = time.time()

        # Initialize target joint values
        # Maintain state for open-loop velocity integration
        print("Waiting for initial joint reading...")
        time.sleep(1)
        curr_joints_deg = robot.get_joint_values()
        if all(j == 0 for j in curr_joints_deg):
             print("[WARNING] Initial joint reading is all zeros. Check connection.")
        
        target_joints_rad = [np.deg2rad(j) for j in curr_joints_deg[:5]] 
        gripper_pos_deg = curr_joints_deg[5]
        print(f"Initial target joints (rad): {target_joints_rad}")

        initial_ee, _ = model.calc_forward_kinematics(target_joints_rad[:5])
        prev_pos = np.array([initial_ee.x, initial_ee.y, initial_ee.z])
        print(f"Start end-effector position (reachable test target): "
              f"{initial_ee.x:.3f} {initial_ee.y:.3f} {initial_ee.z:.3f}")

        print("Waiting for UDP targets on port 5005...")
        while True:
            # Wait for a packet, then handle everything that has arrived.
            # 4-byte packet = latency ping: echo it straight back.
            # 12-byte packet = target: keep only the newest one
            # (targets sent while the arm was moving are stale).
            data, addr = sock.recvfrom(1024)
            target_data = None
            while True:
                if len(data) == 4:
                    sock.sendto(data, addr)
                elif len(data) == 12:
                    target_data, target_addr = data, addr
                try:
                    data, addr = sock.recvfrom(1024, socket.MSG_DONTWAIT)
                except BlockingIOError:
                    break
            if target_data is None:
                continue
            t_recv = time.perf_counter()
            target_pos = list(struct.unpack('<3f', target_data))

            if robot.read_error is not None:
                print("[FATAL] Reader failed:", robot.read_error)
                break
            
            print(f"Moving to target: {target_pos}")
            
            # Get current joints for IK initial guess
            curr_joints_deg = robot.get_joint_values()
            curr_joints_rad = [np.deg2rad(j) for j in curr_joints_deg]
            
            ee = ut.EndEffector()
            ee.x, ee.y, ee.z = target_pos[0], target_pos[1], target_pos[2]
            
            # Use only the first 5 joints (arm) for IK, pass in radians
            # ik_result_rad = model.calc_analytical_inverse_kinematics(ee, curr_joints_rad)
            ik_result_rad = model.calc_numerical_ik(ee, curr_joints_rad[:5])
            t_ik = time.perf_counter()

            # calc_numerical_ik returns all zeros when it finds no solution; moving there would
            # swing the arm to an unplanned pose at full speed (near the floor in the block test)
            if all(j == 0 for j in ik_result_rad):
                print("[WARNING] IK failed (target unreachable?), skipping")
                continue

            # Append gripper (keep current value)
            full_joints_rad = list(ik_result_rad) + [curr_joints_rad[5]]
            
            # Move to the target position
            # Duration determines time, but not speed, so we can use an estimated optimal speed for duration
            optimal_speed = .6 #m/s; measured: servos reach 95% in ~the commanded time down to 0.13 s (9 cm at 0.7 m/s)
            distance = np.linalg.norm(np.array(target_pos) - np.array(prev_pos))
            move_duration = distance / optimal_speed
            # tiny moves would otherwise get ~0 s = "as fast as possible"; measured servos keep up to ~0.13 s
            move_duration = max(move_duration, 0.1)
            robot.set_joint_values(full_joints_rad, duration=move_duration, radians=True)
            t_sent = time.perf_counter()

            # Ack '<3f': IK time, total Pi time (receive -> servo command written), move duration (s).
            # The laptop subtracts Pi time from its round trip to get network time.
            sock.sendto(struct.pack('<3f', t_ik - t_recv, t_sent - t_recv, move_duration), target_addr)
            print(f"Move duration: {move_duration:.2f} s | IK {1000 * (t_ik - t_recv):.1f} ms | "
                  f"Pi total {1000 * (t_sent - t_recv):.1f} ms")
            # Wait for the movement to complete plus a small buffer
            time.sleep(move_duration + (move_duration**2) * .5)
            prev_pos = target_pos


    except KeyboardInterrupt:
        print("\n[INFO] Keyboard Interrupt detected. Initiating shutdown...")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        sock.close()
        robot.shutdown_robot()




if __name__ == "__main__":
    main()
