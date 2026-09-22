import time
import mujoco
import mujoco.viewer
import numpy as np

# 1. Load the model and data environment
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# Define the Triangle Vertices [X, Y, Z]
vertices = [
    np.array([-0.0208, -0.2390, 0.1508]),  # Vertex A
    np.array([-0.1571, 0.0040, 0.3709]),   # Vertex B
    np.array([0.1200, -0.1500, 0.2500]),   # Vertex C
    np.array([-0.0208, -0.2390, 0.1508])   # Return to Vertex A
]

def solve_ik(model, data, target_pos, body_name='Moving_Jaw', max_steps=100, tol=1e-4):
    """Numerical IK solver using the Levenberg-Marquardt damped least-squares method."""
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = data.qpos[:]
    mujoco.mj_fwdPosition(model, ik_data)
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    damping = 0.01
    step_size = 0.4
    
    for _ in range(max_steps):
        current_pos = ik_data.body(body_name).xpos
        error = target_pos - current_pos
        
        if np.linalg.norm(error) < tol:
            break
            
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, ik_data, jac, None, body_id)
        jac_arm = jac[:, :6]
        
        delta_q = jac_arm.T @ np.linalg.inv(jac_arm @ jac_arm.T + damping**2 * np.eye(3)) @ error
        ik_data.qpos[:6] += step_size * delta_q
        
        for j in range(6):
            if model.jnt_limited[j]:
                ik_data.qpos[j] = np.clip(ik_data.qpos[j], model.jnt_range[j][0], model.jnt_range[j][1])
                
        mujoco.mj_fwdPosition(model, ik_data)
        
    return ik_data.qpos[:6].copy()

def move_linear_cartesian(start_pt, goal_pt, duration, viewer):
    start_sim_time = data.time
    elapsed = 0.0
    
    while elapsed < duration and viewer.is_running():
        step_start = time.time()
        
        elapsed = data.time - start_sim_time
        t = min(elapsed, duration)
        
        # 1. Smooth scaling parameter
        s = 10.0 * (t / duration)**3 - 15.0 * (t / duration)**4 + 6.0 * (t / duration)**5
        
        # 2. Perfect linear waypoint
        waypoint_xyz = start_pt + s * (goal_pt - start_pt)
        
        # 3. Calculate target angles via IK
        target_qpos = solve_ik(model, data, waypoint_xyz)
        
        # Direct geometric overwrite to check true math paths
        data.qpos[:6] = target_qpos
        mujoco.mj_fwdPosition(model, data)
        
        viewer.sync()
        data.time += model.opt.timestep
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

def main():
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Simulation running. Visual overlays removed.")
        time.sleep(1.0)
        
        initial_ik = solve_ik(model, data, vertices[0])
        data.qpos[:6] = initial_ik
        mujoco.mj_fwdPosition(model, data)
        
        segment_duration = 3.0
        
        move_linear_cartesian(vertices[0], vertices[1], segment_duration, viewer)
        time.sleep(0.5)
        move_linear_cartesian(vertices[1], vertices[2], segment_duration, viewer)
        time.sleep(0.5)
        move_linear_cartesian(vertices[2], vertices[3], segment_duration, viewer)
        
        print("\nTrajectory finished.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
