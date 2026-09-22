import time
import mujoco
import mujoco.viewer
import numpy as np

# 1. Load the model and data environment
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

def generate_quintic_spline(q_start, q_goal, T):
    """
    Calculates the 5th-degree polynomial coefficients for each joint.
    Returns an array of shape (6, 6) containing coefficients [a0, a1, a2, a3, a4, a5]
    """
    coefficients = []
    for q0, qf in zip(q_start, q_goal):
        a0 = q0
        a1 = 0.0
        a2 = 0.0
        a3 = 10.0 * (qf - q0) / (T ** 3)
        a4 = -15.0 * (qf - q0) / (T ** 4)
        a5 = 6.0 * (qf - q0) / (T ** 5)
        coefficients.append([a0, a1, a2, a3, a4, a5])
    return np.array(coefficients)

def move_to_angles(target_angles, duration, viewer):
    """
    Smoothly moves the arm from its current joint positions to the target_angles
    over a specified duration using a 5th-degree polynomial trajectory.
    """
    print(f"\n[Trajectory] Planning path to: {target_angles} over {duration}s")
    
    # Capture the exact position where the actuators are right now
    start_angles = np.array(data.ctrl[:6])
    
    # Generate the polynomial trajectory coefficients
    coeffs = generate_quintic_spline(start_angles, target_angles, duration)
    
    start_sim_time = data.time
    elapsed = 0.0
    
    while elapsed < duration and viewer.is_running():
        step_start = time.time()
        
        # Calculate elapsed simulation time
        elapsed = data.time - start_sim_time
        t = min(elapsed, duration)
        
        # Calculate the intermediate target position for each joint at time 't'
        current_targets = []
        for j in range(6):
            a = coeffs[j]
            pos_t = a[0] + a[1]*t + a[2]*(t**2) + a[3]*(t**3) + a[4]*(t**4) + a[5]*(t**5)
            current_targets.append(pos_t)
            
        # Write the intermediate targets into the position actuators
        data.ctrl[:6] = current_targets
        
        # Advance physics simulation step
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # Enforce real-time loop synchronization
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
    print("[Trajectory] Movement Complete.")
    
    # Force a forward position pass to update kinematics matrices immediately
    mujoco.mj_fwdPosition(model, data)
    ee_position = data.body('Moving_Jaw').xpos
    print(f"-> Arrived at End-Effector XYZ: [{ee_position[0]:.4f}, {ee_position[1]:.4f}, {ee_position[2]:.4f}]")

def main():
    # Force initialize the simulation state to the XML home profile keyframe
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation actively running. Executing sequence...")
        time.sleep(1.0) # Short buffer for visualizer initialization
        
        # Define the exact 3-pose cycle
        pose_1 = [1.27, -2.62, 0.837, -0.614, 0.0, 1.18]        
        pose_2 = [-0.768, -3.11, 0.92, -0.797, -1.03, -0.174]   
        pose_3 = [0.0, -1.57, 1.57, 1.57, -1.57, 0.0]           
        
        # Step 1: Move to Pose 1 over 3.5 seconds
        move_to_angles(target_angles=pose_1, duration=3.5, viewer=viewer)
        time.sleep(1.5) # Hold pose briefly
        
        # Step 2: Move to Pose 2 over 4.0 seconds
        move_to_angles(target_angles=pose_2, duration=4.0, viewer=viewer)
        time.sleep(1.5)
        
        # Step 3: Return to Pose 3 (Home) over 3.0 seconds
        move_to_angles(target_angles=pose_3, duration=3.0, viewer=viewer)
        
        # Keep visualizer open for interactive observation
        print("\nAll sequence movements finalized.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
