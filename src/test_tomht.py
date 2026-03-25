import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from process_centroid_to_tracks import TOMHTTracker 


def generate_scenarios():
    """Generates synthetic (x, y) measurements for 4 different testing scenarios."""
    scenarios = {}
    
    # ---------------------------------------------------------
    # Scenario 1: Crossing Paths (Tests KD-Tree Clustering)
    # Two targets start at opposite corners and cross exactly in the middle
    # ---------------------------------------------------------
    frames = 20
    s1_data = {}
    for t in range(frames):
        # Target A moves (0,0) to (100,100)
        ta = [t * 5.0, t * 5.0]
        # Target B moves (0,100) to (100,0)
        tb = [t * 5.0, 100.0 - (t * 5.0)]
        s1_data[t] = np.array([ta, tb])
    scenarios["1. Crossing Paths"] = s1_data

    # ---------------------------------------------------------
    # Scenario 2: Y-Split Clutter (Tests MHT Branching/Scoring)
    # A straight track encounters a false clutter point that veers off
    # ---------------------------------------------------------
    s2_data = {}
    for t in range(frames):
        meas = [[t * 5.0, 50.0]]  # True target moves straight right
        # Introduce clutter between frames 5 and 9
        if 5 <= t <= 9:
            clutter_y = 50.0 + ((t - 5) * 10.0) # Veers sharply upward
            meas.append([t * 5.0, clutter_y])
        s2_data[t] = np.array(meas)
    scenarios["2. Y-Split Clutter"] = s2_data

    # ---------------------------------------------------------
    # Scenario 3: Vanishing Target (Tests Pruning / Missed Detections)
    # Target moves normally for 10 frames, then disappears entirely
    # ---------------------------------------------------------
    s3_data = {}
    for t in range(frames):
        if t < 10:
            s3_data[t] = np.array([[t * 5.0, 20.0]])
        else:
            s3_data[t] = np.array([]) # Empty array, simulating total occlusion
    scenarios["3. Vanishing Target"] = s3_data

    # ---------------------------------------------------------
    # Scenario 4: Sine Wave (Tests Kalman Filter Dynamics/Process Noise)
    # Target weaves up and down, requiring the filter to handle turns
    # ---------------------------------------------------------
    s4_data = {}
    for t in range(frames):
        x = t * 5.0
        y = 50.0 + 30.0 * np.sin(x / 10.0)
        s4_data[t] = np.array([[x, y]])
    scenarios["4. Sine Wave Dynamics"] = s4_data

    return scenarios

def run_test_and_plot():
    scenarios = generate_scenarios()
    
    # Create a 2x2 plotting grid
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    
    for i, (name, meas_dict) in enumerate(scenarios.items()):
        ax = axs[i]
        
        # Instantiate a fresh tracker for each scenario
        # We use a wide gate distance and high process noise so the filter connects the dots easily
        tracker = TOMHTTracker(dt=1.0, gate_distance=30.0, max_misses=5)
        # Override KF noises to be more forgiving for these tests
        tracker.kf.process_noise_std = 50.0 
        tracker.kf.measurement_noise_std = 15.0
        
        # We will record the outputs to plot continuous lines
        track_history = []
        raw_x, raw_y = [], []
        
        for t, measurements in meas_dict.items():
            # Store raw measurements for background plotting
            if len(measurements) > 0:
                raw_x.extend(measurements[:, 0])
                raw_y.extend(measurements[:, 1])
                
            # Step the tracker
            active_tracks = tracker.step(measurements)
            
            # Record track states
            for trk in active_tracks:
                if trk.age >= 2: # Only record confirmed tracks
                    track_history.append({
                        'time': t,
                        'id': trk.track_id,
                        'x': trk.best_state[0],
                        'y': trk.best_state[1]
                    })
        
        # 1. Plot raw measurements (Gray Dots)
        ax.scatter(raw_x, raw_y, c='lightgray', s=80, label='Raw Measurements', zorder=1)
        
        # 2. Plot Tracked Paths (Colored Lines)
        if len(track_history) > 0:
            df = pd.DataFrame(track_history)
            for trk_id, grp in df.groupby('id'):
                ax.plot(grp['x'], grp['y'], marker='o', markersize=4, linewidth=2, 
                        label=f'Track {trk_id} (len={len(grp)})', zorder=2)
                
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)
        
    plt.tight_layout()
    plt.savefig("tomht_synthetic_tests.png", dpi=150)
    print("Tests complete! Saved results to 'tomht_synthetic_tests.png'")
    plt.show()

if __name__ == "__main__":
    run_test_and_plot()