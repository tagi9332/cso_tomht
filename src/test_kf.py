import numpy as np
import matplotlib.pyplot as plt

# Import your class (adjust the import path if necessary)
from utils.kalman_filter import KalmanFilter2D

def generate_synthetic_data(num_frames=20, dt=1.0, noise_std=2.0):
    """Generates a target moving at constant velocity with noisy measurements."""
    # True initial state: px=10, py=10, vx=5, vy=3
    true_x = np.array([10.0, 10.0, 5.0, 3.0])
    
    # State transition matrix for generating ground truth
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    
    true_positions = []
    measurements = []
    
    current_x = true_x
    for _ in range(num_frames):
        # 1. Record true position
        true_positions.append(current_x[0:2].copy())
        
        # 2. Add Gaussian noise to simulate a real sensor
        noise = np.random.normal(0, noise_std, size=2)
        measurements.append(current_x[0:2] + noise)
        
        # 3. Step truth forward
        current_x = F @ current_x
        
    return np.array(true_positions), np.array(measurements)

def test_kalman_filter():
    print("--- Starting Kalman Filter 2D Test ---")
    
    # 1. Setup Data and Filter
    dt = 1.0
    kf = KalmanFilter2D(dt=dt, process_noise_std=0.5, measurement_noise_std=2.0)
    true_pos, measurements = generate_synthetic_data(num_frames=25, dt=dt)
    
    # 2. Storage for plotting
    filtered_states = []
    
    # 3. Initialize the filter with the first measurement
    z0 = measurements[0]
    x, P = kf.initialize(z0)
    filtered_states.append(x)
    print(f"Frame 0: Initialized at {z0}")
    
    # 4. Run the predict-update cycle
    for i, z in enumerate(measurements[1:], start=1):
        # Predict
        x_pred, P_pred = kf.predict(x, P)
        
        # Optional: Check Mahalanobis distance just to see it work
        dist = kf.mahalanobis_distance(x_pred, P_pred, z)
        
        # Update
        x, P, log_l = kf.update(x_pred, P_pred, z)
        filtered_states.append(x)
        
        print(f"Frame {i}: Measured={z.round(2)}, Estimated={x[0:2].round(2)}, Vel={x[2:4].round(2)}, M-Dist={dist:.2f}")

    filtered_states = np.array(filtered_states)
    
    # 5. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(true_pos[:, 0], true_pos[:, 1], 'g--', label='Ground Truth', linewidth=2)
    plt.scatter(measurements[:, 0], measurements[:, 1], c='red', marker='x', label='Noisy Measurements (z)')
    plt.plot(filtered_states[:, 0], filtered_states[:, 1], 'b-', label='KF Estimate (x)', linewidth=2)
    
    plt.title("2D Kalman Filter Tracking a Single Target with Simulated Data")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("kalman_filter_test.png", dpi=150)
if __name__ == "__main__":
    test_kalman_filter()