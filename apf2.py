import mujoco
import mujoco.viewer
import numpy as np

# Load the main scene file containing the obstacle and target mocap markers
model = mujoco.MjModel.from_xml_path('scene2.xml')
data = mujoco.MjData(model)

# -----------------------------------------------------------------
# APF Control Hyperparameters
# -----------------------------------------------------------------
K_ATT = 2000.0      # Target attraction stiffness
K_REP = 10.0        # Obstacle repulsion gain per keypoint
RHO_0 = 0.10        # Safety field radius around obstacle (meters)
DAMPING = 7.0       # Joint-space damping coefficient to prevent oscillations

# -----------------------------------------------------------------
# Operational Space Tracking Identifiers
# -----------------------------------------------------------------
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_mocap")
obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle_mocap")
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")

# Identify body segments where specific point-offsets will be checked
upper_arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Upper_Arm")
lower_arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Lower_Arm")
wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Wrist_Pitch_Roll")
fixed_jaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw") # <-- Added gripper body

def compute_apf_avoidance(model, data):
    tau = np.zeros(model.nv)
    
    # Extract positions of operational landmarks
    TARGET_POS = data.xpos[target_id]
    OBSTACLE_POS = data.xpos[obstacle_id]
    x_ee = data.site_xpos[ee_site_id]
    
    # -----------------------------------------------------------------
    # 1. TARGET ATTRACTION (End-Effector Site)
    # -----------------------------------------------------------------
    f_att = -K_ATT * (x_ee - TARGET_POS)
    
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id)
    tau += jac_p.T @ f_att

    # -----------------------------------------------------------------
    # 2. KEYPOINT OBSTACLE REPULSION
    # -----------------------------------------------------------------
    # Added a critical protection point explicitly on the main gripper claw body
    keypoints = [
        {"body_id": upper_arm_id, "local_pos": np.array([0.0, 0.05, 0.0])},   # Midpoint of Upper Arm
        {"body_id": upper_arm_id, "local_pos": np.array([0.0, 0.11, 0.0])},   # Near Elbow Joint
        {"body_id": lower_arm_id, "local_pos": np.array([0.0, 0.0, 0.06])},   # Midpoint of Lower Arm
        {"body_id": lower_arm_id, "local_pos": np.array([0.0, 0.0, 0.13])},   # Near Wrist Assembly
        {"body_id": wrist_id,     "local_pos": np.array([0.0, -0.03, 0.0])},  # Gripper Base region
        {"body_id": fixed_jaw_id, "local_pos": np.array([0.01, -0.06, 0.0])}, # <-- NEW: Gripper Upper Claw Point
    ]
    
    for kp in keypoints:
        b_id = kp["body_id"]
        
        # Extract the body's global position and 3x3 orientation matrix
        body_xpos = data.xpos[b_id]
        body_xmat = data.xmat[b_id].reshape(3, 3)
        
        # Manually compute local-to-global transformation to bypass strict C bindings
        x_point = body_xpos + body_xmat @ kp["local_pos"]
        
        # Calculate spatial vector and distance from the target point to obstacle
        vec = x_point - OBSTACLE_POS
        dist = np.linalg.norm(vec)
        
        # If the point penetrates the security bubble, accumulate repulsive torques
        if 0.001 < dist < RHO_0:
            # APF Force curve formulation
            f_rep_mag = K_REP * (1.0 / dist - 1.0 / RHO_0) * (1.0 / (dist ** 2))
            f_rep = f_rep_mag * (vec / dist)
            
            # Compute the point Jacobian for this specific link offset position
            jac_p_point = np.zeros((3, model.nv))
            jac_r_point = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac_p_point, jac_r_point, x_point, b_id)
            
            # Project the point repulsion force directly into your control forces
            tau += jac_p_point.T @ f_rep

    # -----------------------------------------------------------------
    # 3. STABILIZATION
    # -----------------------------------------------------------------
    tau -= DAMPING * data.qvel  

    return tau

# Core simulation execution loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    while viewer.is_running():
        # Compute joint forces balancing attraction and keypoint obstacle deflection
        tau_control = compute_apf_avoidance(model, data)
        
        # Direct torque injection bypassing the passive PID actuators
        data.qfrc_applied[:] = tau_control
        
        mujoco.mj_step(model, data)
        viewer.sync()