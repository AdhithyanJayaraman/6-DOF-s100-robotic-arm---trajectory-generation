import time
import mujoco
import mujoco.viewer
import numpy as np

# 1. Load the model (points to your main scene file)
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 2. Define your ONE AND ONLY target joint angle set (in Radians)
# QUICK REFERENCE JOINT LIMITS:
# [0] Rotation    : -1.92   to  1.92
# [1] Pitch       : -3.32   to  0.174
# [2] Elbow       : -0.174  to  3.14
# [3] Wrist_Pitch : -1.66   to  1.66
# [4] Wrist_Roll  : -2.79   to  2.79
# [5] Jaw         : -0.174  to  1.75
single_target_pose = [1.9, -3.3, 3.1, 1.5, 2.5, 1.]

def main():
    # Set the actuator controls to our target pose immediately before starting
    data.ctrl[:len(single_target_pose)] = single_target_pose

    # Force MuJoCo to calculate Forward Kinematics right now for the initial state
    mujoco.mj_fwdPosition(model, data)
    
    # Extract the global 3D position [X, Y, Z] of the end-effector
    # We use 'Moving_Jaw' since it's the tip of your robot arm
    ee_position = data.body('Moving_Jaw').xpos
    print("=" * 50)
    print("PREDICTED END-EFFECTOR INITIAL POSITION (FK):")
    print(f"X: {ee_position[0]:.4f} meters")
    print(f"Y: {ee_position[1]:.4f} meters")
    print(f"Z: {ee_position[2]:.4f} meters")
    print("=" * 50)

    # Launch the passive visualizer window
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nSimulation started. Close the viewer window to stop.")
        
        # Simple tracking variable to limit how often we spam the terminal
        last_print_time = 0.0
        
        while viewer.is_running():
            step_start = time.time()
            
            # Re-apply the target pose to ensure the actuators hold it tightly
            data.ctrl[:len(single_target_pose)] = single_target_pose
            
            # Step the physical engine forward by one timestep (computes FK automatically)
            mujoco.mj_step(model, data)
            
            # Print the real-time position every 0.5 seconds as it settles
            if data.time - last_print_time > 0.5:
                current_ee_pos = data.body('Moving_Jaw').xpos
                print(f"Time: {data.time:.1f}s | EE Live Position -> X: {current_ee_pos[0]:.3f}, Y: {current_ee_pos[1]:.3f}, Z: {current_ee_pos[2]:.3f}")
                last_print_time = data.time
            
            # Refresh the visualizer screen
            viewer.sync()
            
            # Maintain real-time execution pace
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
