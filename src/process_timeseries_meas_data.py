import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any

# Local imports
from utils.kdtree_association import KDTreeAssociation
from track import Track, Hypothesis

def print_tracking_summary(association_history, truth_df, tracked_df, match_threshold=3.0):
    """
    Analyzes the tracking history and dataframes to print a performance summary.
    """
    # 1. Count internal cluster events
    new_clusters = 0
    merges_or_conflicts = 0
    simple_updates = 0
    
    for frame in association_history:
        for cluster in frame['clusters']:
            t_idxs = cluster['track_indices']
            m_idxs = cluster['meas_indices']
            
            if len(t_idxs) == 0:
                # Unassociated measurements turning into new tracks
                new_clusters += len(m_idxs) 
            elif len(t_idxs) > 1 or len(m_idxs) > 1:
                # Multiple tracks or multiple measurements overlapping
                merges_or_conflicts += 1
            elif len(t_idxs) == 1 and len(m_idxs) == 1:
                # Perfect 1-to-1 association
                simple_updates += 1

    # 2. Calculate Truth vs. Tracked Accuracy
    true_ids = 0     # True Positives
    false_ids = 0    # False Positives (Ghost tracks)
    missed_det = 0   # False Negatives (Dropped tracks)
    
    time_steps = sorted(truth_df['time'].unique())
    
    for t in time_steps:
        truth_t = truth_df[truth_df['time'] == t][['x', 'y']].to_numpy()
        
        # If tracker missed a frame entirely, handle gracefully
        if t in tracked_df['time'].values:
            track_t = tracked_df[tracked_df['time'] == t][['x', 'y']].to_numpy()
        else:
            track_t = np.array([])
        
        matched_truth_indices = set()
        
        # Check every tracked point against the truth
        for trk in track_t:
            if len(truth_t) == 0:
                false_ids += 1
                continue
                
            # Find the closest ground truth point
            distances = np.linalg.norm(truth_t - trk, axis=1)
            best_match_idx = np.argmin(distances)
            
            if distances[best_match_idx] <= match_threshold and best_match_idx not in matched_truth_indices:
                true_ids += 1
                matched_truth_indices.add(best_match_idx)
            else:
                false_ids += 1
                
        # Any truth points that weren't matched are misses
        missed_det += (len(truth_t) - len(matched_truth_indices))

    # 3. Print the Summary
    print("\n" + "="*40)
    print("       TRACKING PIPELINE SUMMARY")
    print("="*40)
    print(f"Total Time Steps Processed : {len(time_steps)}")
    print(f"Total Ground Truth Points  : {len(truth_df)}")
    print(f"Total Tracked Points       : {len(tracked_df)}")
    print("-" * 40)
    print(" EVENT COUNTERS (From KD-Tree)")
    print("-" * 40)
    print(f" New Track Initiations     : {new_clusters}")
    print(f" Clean 1-to-1 Updates      : {simple_updates}")
    print(f" Merges/Conflicts Handled  : {merges_or_conflicts}")
    print("-" * 40)
    print(" ACCURACY ESTIMATION")
    print("-" * 40)
    print(f" True Identifications (TP) : {true_ids}")
    print(f" False Identifications (FP): {false_ids}")
    print(f" Missed Detections (FN)    : {missed_det}")
    print("="*40 + "\n")


# --- 2. Static Plotting Function ---
def plot_static_tracks(truth_df, tracked_df, output_file="static_tracks.png"):
    """Plots the ground truth vs the tracker output on a 2D scatter plot."""
    plt.figure(figsize=(10, 6))
    
    # Plot Ground Truth (from the raw CSV)
    for truth_id in truth_df['id'].unique():
        t_data = truth_df[truth_df['id'] == truth_id]
        plt.plot(t_data['x'], t_data['y'], 'o--', alpha=0.3, label=f'Truth ID {truth_id}')
        
    # Plot Tracker Output
    for track_id in tracked_df['id'].unique():
        trk_data = tracked_df[tracked_df['id'] == track_id]
        plt.plot(trk_data['x'], trk_data['y'], 'x-', linewidth=2, label=f'Track ID {track_id}')
        
    plt.title("Tracker Output vs Ground Truth")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved static plot to {output_file}")


# --- 3. Animation Function ---
def animate_tracks(tracked_df, output_file="tracked_animation.gif"):
    """Creates a GIF animation of the tracker output over time."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get plot limits
    ax.set_xlim(tracked_df['x'].min() - 5, tracked_df['x'].max() + 5)
    ax.set_ylim(tracked_df['y'].min() - 5, tracked_df['y'].max() + 5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    time_steps = sorted(tracked_df['time'].unique())
    
    # Dictionary to hold the line objects for each track
    lines = {}
    colors = plt.cm.tab10.colors # Use standard color map
    
    def update(frame_time):
        ax.set_title(f"Tracker Output - Time: {frame_time:.2f}")
        
        # Get all data up to the current frame
        current_data = tracked_df[tracked_df['time'] <= frame_time]
        
        # Update each track's line
        for track_id in current_data['id'].unique():
            trk_history = current_data[current_data['id'] == track_id]
            
            if track_id not in lines:
                # Create a new line for a new track
                color = colors[int(track_id) % len(colors)]
                line, = ax.plot([], [], 'o-', color=color, label=f'Track {track_id}')
                lines[track_id] = line
                ax.legend(loc='upper left')
                
            # Set the updated data
            lines[track_id].set_data(trk_history['x'], trk_history['y'])
            
        return list(lines.values())

    # Generate the animation
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=time_steps, 
        interval=100, # 100ms per frame
        blit=False
    )
    
    ani.save(output_file, writer='pillow')
    print(f"Saved animation to {output_file}")

def process_timeseries_meas_data(meas_file: str, gate_distance: float) -> tuple[List[Dict], pd.DataFrame]:
    """Process time-series measurement data for multi-target tracking."""
    # Load measurement data
    meas_df = pd.read_csv(meas_file)
    time_steps = sorted(meas_df['time'].unique())

    # Initialize associator with merge_clusters=True!
    assoc = KDTreeAssociation(gate_distance=gate_distance, merge_clusters=True)

    # Track management
    active_tracks = {}
    next_track_id = 0

    # Initialize histories
    association_history = []
    track_history = []

    # Process each time step
    for t in time_steps:
        # Get measurements for this time step
        meas_t = meas_df[meas_df['time'] == t]
        measurements = meas_t[['x', 'y']].to_numpy()

        # Initialize arrays for k-d tree association
        predicted_positions = []
        track_ids = []

        # Predict positions for existing tracks
        for track_id, state in active_tracks.items():
            # Simple constant velocity prediction
            pred_x = state[0] + state[2]  # px + vx
            pred_y = state[1] + state[3]  # py + vy

            predicted_positions.append([pred_x, pred_y])
            track_ids.append(track_id)

        predicted_positions = np.array(predicted_positions)

        # Call k-d tree CLUSTER to get independent groups of tracks and measurements
        if len(predicted_positions) > 0 and len(measurements) > 0:
            clusters = assoc.cluster(predicted_positions, measurements)
        elif len(measurements) > 0:
            # No tracks yet, all measurements are unassociated
            clusters = [{"track_indices": [], "meas_indices": list(range(len(measurements)))}]
        else:
            clusters = []

        # Store the associations for this time step
        association_history.append({
            'time': t,
            'clusters': clusters,
            'track_ids': track_ids,
            'measurements': measurements
        })

        # --- PROCESS CLUSTERS ---
        associated_track_indices = set()

        for cluster in clusters:
            trk_idxs = cluster["track_indices"]
            meas_idxs = cluster["meas_indices"]
            
            associated_track_indices.update(trk_idxs)

            # NEW CLUSTER FORMATION (Unassociated Measurements)
            if len(trk_idxs) == 0:
                for m_idx in meas_idxs:
                    z = measurements[m_idx]
                    # Start a new track with 0 velocity
                    active_tracks[next_track_id] = [z[0], z[1], 0.0, 0.0]
                    next_track_id += 1

            # SIMPLE UPDATE (1 Track, 1 Measurement)
            elif len(trk_idxs) == 1 and len(meas_idxs) == 1:
                t_idx = trk_idxs[0]
                m_idx = meas_idxs[0]
                actual_track_id = track_ids[t_idx]
                z = measurements[m_idx]
                
                # Update track state based on measurement
                old_x, old_y, _, _ = active_tracks[actual_track_id]
                new_vx = z[0] - old_x
                new_vy = z[1] - old_y
                active_tracks[actual_track_id] = [z[0], z[1], new_vx, new_vy]

            # COMBINED CLUSTERS (Multiple overlapping tracks/measurements)
            else:
                for t_idx in trk_idxs:
                    actual_track_id = track_ids[t_idx]
                    pred_pos = predicted_positions[t_idx]
                    
                    # Find the closest measurement in this specific cluster
                    best_m_idx = None
                    min_dist = float('inf')
                    for m_idx in meas_idxs:
                        dist = np.linalg.norm(pred_pos - measurements[m_idx])
                        if dist < min_dist:
                            min_dist = dist
                            best_m_idx = m_idx
                            
                    # Update track with the best local measurement
                    if best_m_idx is not None:
                        z = measurements[best_m_idx]
                        old_x, old_y, _, _ = active_tracks[actual_track_id]
                        active_tracks[actual_track_id] = [z[0], z[1], z[0]-old_x, z[1]-old_y]

        # HANDLE MISSED DETECTIONS
        for i, t_id in enumerate(track_ids):
            if i not in associated_track_indices:
                # "Coast" the track using its predicted position
                old_x, old_y, vx, vy = active_tracks[t_id]
                active_tracks[t_id] = [old_x + vx, old_y + vy, vx, vy]
                
        # --- RECORD HISTORY HERE ---
        # Now that all tracks have been updated or coasted for time `t`, record their state
        for t_id, state in active_tracks.items():
            track_history.append({
                'time': t,
                'id': t_id,
                'x': state[0],
                'y': state[1],
                'vx': state[2],
                'vy': state[3]
            })

    # Return the association history, and format the track history as a DataFrame for easy plotting
    return association_history, pd.DataFrame(track_history)


if __name__ == "__main__":
    # Set the measurement file path
    meas_file = "data/Curving_m.csv"
    
    # Load the ground truth just for plotting comparisons
    truth_df = pd.read_csv(meas_file)
    
    # Unpack results
    print("Running tracker...")
    gate_distance = 5.0  # Example gating distance for association
    association_history, tracked_df = process_timeseries_meas_data(meas_file, gate_distance=gate_distance)

    # Set output directory for visualizations
    output_dir = "results/"
    
    # Generate visualizations
    print("Generating static plot...")
    plot_static_tracks(truth_df, tracked_df, output_dir + "static_comparison.png")
    
    print("Generating animation...")
    animate_tracks(tracked_df, output_dir + "tracker_output.gif")

    # Summarize tracking performance
    # Run the summary block
    print_tracking_summary(
        association_history, 
        truth_df, 
        tracked_df, 
        match_threshold=gate_distance
    )
    
    print("Done!")