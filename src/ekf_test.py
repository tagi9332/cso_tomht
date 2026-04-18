import os, sys
import numpy as np
from dataclasses import dataclass, field

# Set project root and add to sys.path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.prediction_filters.ekf_class import CWH_AnglesOnly_EKF

# --- Mock Configuration Classes ---
@dataclass
class MockCoeffs:
    mean_motion: float
    focal_length: float
    nominal_distance: float

@dataclass
class MockEKFOptions:
    coeffs: MockCoeffs
    method: str = "SNC"
    frame_type: str = "RIC"
    # FIXED: Use default_factory for mutable numpy arrays
    q_cont: np.ndarray = field(default_factory=lambda: np.eye(3) * 1e-2)
    threshold: float = 10.0

def run_ekf_test():
    print("=== EKF FULL NMC ORBIT TEST ===")
    
    # 1. Orbital Setup (FIXED: Defined these before using them)
    mean_motion = 0.000072921  # GEO rad/s
    orbital_period = 2 * np.pi / mean_motion  # ~86164 seconds (1 day)
    dt = 60.0  # 1-minute time steps for the simulation
    total_steps = int(orbital_period / dt)
    
    ekf_options = MockEKFOptions(
        coeffs=MockCoeffs(
            mean_motion=mean_motion, 
            focal_length=4.0,
            nominal_distance=1000.0  # 1km starting depth guess
        )
    )
    
    # FIXED: R_matrix: 4.0 var for pixels, 1e8 for pseudo-range (Ignore depth!)
    R_matrix = np.diag([4.0, 4.0, 1e8]) 
    
    # Initialize EKF
    ekf = CWH_AnglesOnly_EKF(dt=dt, ekf_options=ekf_options, R_matrix=R_matrix)
    
    # 2. Generate True NMC Initial State
    x_0 = 500.0  # 500m radial amplitude
    y_0 = 0.0    # Start crossing the y-axis
    z_0 = 200.0  # 200m cross-track amplitude
    
    # Enforce NMC constraints
    vx_0 = 0.5 * y_0 * mean_motion
    vy_0 = -2.0 * x_0 * mean_motion
    vz_0 = 0.0  # Max cross-track amplitude means zero cross-track velocity here
    
    true_state = np.array([x_0, y_0, z_0, vx_0, vy_0, vz_0])
    
    # 3. Get True STM to propagate the true state perfectly
    Phi_true = ekf.compute_cwh_stm(dt)
    
    print(f"Simulating {total_steps} steps (dt={dt}s) for one full GEO orbit...")
    
    # 4. Main Simulation Loop
    for step in range(total_steps):
        # --- A. Propagate True State ---
        true_state = Phi_true @ true_state
        
        # --- B. Generate Noisy Measurement ---
        # Get perfect focal plane coordinates (u, v) and pseudo-range (x)
        z_perfect = ekf.compute_h_nonlinear(true_state) 
        
        # Add pixel noise (std dev = 2.0 -> var = 4.0)
        u_noise = np.random.normal(0, 2.0)
        v_noise = np.random.normal(0, 2.0)
        
        # Hardcode the pseudo-depth to 0.0 as your tracker does
        z_noisy = np.array([z_perfect[0] + u_noise, z_perfect[1] + v_noise, 0.0])
        
        # --- C. EKF Operations ---
        if step == 0:
            # Give it a warm start for the depth guess (450 instead of 0.0)
            z_init = np.array([z_perfect[0], z_perfect[1], 450.0])
            est_state, P_est = ekf.initialize(z_init)
            
            # Keep the large covariance so it still learns the velocity
            P_est = np.diag([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
            print("\n[Step 0] EKF Initialized.")
            continue
            
        # Predict
        x_pred, P_pred = ekf.predict(est_state, P_est)
        
        # Update
        est_state, P_est, _ = ekf.update(x_pred, P_pred, z_noisy)
        
        # --- D. Telemetry Printouts ---
        # Print progress at 25%, 50%, 75%, and 100% of the orbit
        if (step + 1) % (total_steps // 4) == 0:
            pos_err = np.linalg.norm(true_state[:3] - est_state[:3])
            vel_err = np.linalg.norm(true_state[3:] - est_state[3:])
            print(f"\n[Step {step + 1}/{total_steps}] - {(step+1)/total_steps * 100:.0f}% of Orbit")
            print(f"  True Pos (x,y,z): [{true_state[0]:.1f}, {true_state[1]:.1f}, {true_state[2]:.1f}]")
            print(f"  Est  Pos (x,y,z): [{est_state[0]:.1f}, {est_state[1]:.1f}, {est_state[2]:.1f}]")
            print(f"  3D Position Error: {pos_err:.2f} meters")
            print(f"  3D Velocity Error: {vel_err:.4f} m/s")

if __name__ == "__main__":
    run_ekf_test()