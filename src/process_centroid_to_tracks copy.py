import numpy as np
import pandas as pd
from typing import List, Dict
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Import local modules
from utils.kdtree_association import KDTreeAssociation
# from kf import KalmanFilter2D
from utils.kalman_filter import KalmanFilter2D
from src.track import Track, Hypothesis

class TOMHTTracker:
    def __init__(
        self, 
        dt: float = 1.0, 
        gate_distance: float = 3.0,
        max_misses: int = 5,
        max_hypotheses: int = 5
    ):
        self.kf = KalmanFilter2D(dt=dt, process_noise_std=1.0, measurement_noise_std=2.0)
        self.assoc = KDTreeAssociation(gate_distance=gate_distance, merge_clusters=True)
        
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.max_misses = max_misses
        self.max_hypotheses = max_hypotheses

    def _merge_duplicate_tracks(self, merge_distance: float = 3.0):
        """Finds tracks that are tracking the exact same target and deletes the younger one."""
        tracks_to_delete = set()
        track_list = list(self.active_tracks.values())
        
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                t1 = track_list[i]
                t2 = track_list[j]
                
                # Skip if already marked for deletion
                if t1.track_id in tracks_to_delete or t2.track_id in tracks_to_delete:
                    continue
                    
                # Calculate Euclidean distance between their best current states
                dist = np.linalg.norm(t1.best_state[0:2] - t2.best_state[0:2])
                
                # If they are practically on top of each other
                if dist < merge_distance:
                    # Keep the track that has been alive longer (more established history)
                    if t1.age >= t2.age:
                        tracks_to_delete.add(t2.track_id)
                    else:
                        tracks_to_delete.add(t1.track_id)
                        
        # Prune the duplicates
        for t_id in tracks_to_delete:
            del self.active_tracks[t_id]

    def step(self, measurements: np.ndarray) -> List[Track]:
        """Process a single frame of measurements."""
        measurements = np.atleast_2d(measurements) if len(measurements) > 0 else np.array([])
        
        # 1. PREDICT all hypotheses for all active tracks
        predicted_hyps_map = {} # track_id -> list of (x_pred, P_pred)
        best_positions = []
        track_list = list(self.active_tracks.values())
        
        for track in track_list:
            track.age += 1
            preds = []
            for hyp in track.hypotheses:
                x_p, P_p = self.kf.predict(hyp.state, hyp.covariance)
                preds.append((x_p, P_p))
            predicted_hyps_map[track.track_id] = preds
            
            # Use the best hypothesis position for spatial clustering
            best_positions.append(track.best_state[0:2]) 
            
        best_positions = np.array(best_positions)

        # 2. CLUSTER tracks and measurements
        if len(best_positions) > 0 and len(measurements) > 0:
            clusters = self.assoc.cluster(best_positions, measurements)

            # # Debug print
            # print(f"Frame Tracks: {len(track_list)}, Measurements: {len(measurements)}")
            # print(f"Clusters formed: {clusters}")

        elif len(measurements) > 0:
            clusters = [{"track_indices": [], "meas_indices": list(range(len(measurements)))}]
        else:
            clusters = [{"track_indices": list(range(len(track_list))), "meas_indices": []}]

        # 3. EXPAND HYPOTHESES per cluster
        for cluster in clusters:
            t_idxs = cluster["track_indices"]
            m_idxs = cluster["meas_indices"]
            
            # Case A: Unassociated measurements -> New Tracks
            if len(t_idxs) == 0:
                for m_idx in m_idxs:
                    self._initiate_track(measurements[m_idx])
                continue
                
            # Case B: Update existing tracks in the cluster
            for t_idx in t_idxs:
                track = track_list[t_idx]
                predicted_hyps = predicted_hyps_map[track.track_id]
                measurement_updates = []
                
                # CHANGED: Zip the original hypotheses with the predictions so we can reference the previous state
                for hyp, (x_pred, P_pred) in zip(track.hypotheses, predicted_hyps):
                    hyp_updates = []
                    
                    # Option 1: Missed Detection (Null Hypothesis)
                    hyp_updates.append((None, x_pred, P_pred, track.miss_log_likelihood))
                    
                    # Option 2: Associate with a gated measurement
                    for m_idx in m_idxs:
                        z = measurements[m_idx]
                        
                        # Use Mahalanobis distance to strictly gate based on covariance
                        dist = self.kf.mahalanobis_distance(x_pred, P_pred, z)
                        
                        if dist < 5.0: 
                            # --- NEW: MINIMUM VELOCITY GATE ---
                            # Calculate Euclidean distance between the PREVIOUS state and the NEW measurement
                            prev_pos = hyp.state[0:2]
                            pixel_movement = np.linalg.norm(z - prev_pos)
                            
                            # Only branch if the measurement actually moved (e.g., > 1.5 pixels)
                            if pixel_movement >= 1.5:
                                x_upd, P_upd, log_ll = self.kf.update(x_pred, P_pred, z)
                                hyp_updates.append((m_idx, x_upd, P_upd, log_ll))
                            # ----------------------------------
                            
                    measurement_updates.append(hyp_updates)
                
                # Apply the branch-and-prune step
                track.expand_hypotheses(predicted_hyps, measurement_updates)
                track.normalise_scores()
                
                # Update track management stats based on the new best hypothesis
                if track.best_hypothesis.meas_index is None:
                    track.consecutive_misses += 1
                else:
                    track.consecutive_misses = 0

        # 4. PRUNE dead and stationary tracks
        dead_track_ids = []
        for t_id, t in self.active_tracks.items():
            # Condition A: Too many consecutive missed detections
            if t.consecutive_misses >= self.max_misses:
                dead_track_ids.append(t_id)
                continue
                
            # Condition B: Track is stationary
            # Wait until the track has existed for 3 frames so the KF velocity has time to settle
            if t.age >= 3:
                vx, vy = t.best_state[2], t.best_state[3]
                speed = np.sqrt(vx**2 + vy**2)
                
                # If it's moving less than n pixels per frame, it's noise/stars. Kill it.
                if speed < 10:
                    dead_track_ids.append(t_id)

        for t_id in dead_track_ids:
            del self.active_tracks[t_id]
            
        # 5. MERGE duplicate tracks (Track Coalescence)
        self._merge_duplicate_tracks(merge_distance=5.0)
                        
        return list(self.active_tracks.values())
        

    def _initiate_track(self, z: np.ndarray):
        """Helper to spawn a brand new track."""
        x0, P0 = self.kf.initialize(z)
        new_track = Track(
            track_id=self.next_track_id,
            initial_state=x0,
            initial_covariance=P0,
            max_hypotheses=self.max_hypotheses
        )
        self.active_tracks[self.next_track_id] = new_track
        self.next_track_id += 1


def process_pipeline_data(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Map your columns if you haven't already
    df = df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})
    time_steps = sorted(df['time'].unique())
    
    tracker = TOMHTTracker(dt=1.0, gate_distance=120, max_misses=3, max_hypotheses=5)
    
    # Dictionary to store the most up-to-date history for every track ever seen
    # Format: {track_id: [list of row dicts]}
    all_histories = {} 
    
    for t in time_steps:
        # Extract [x, y] coordinates for this frame
        meas_t = df[df['time'] == t][['x', 'y']].to_numpy()
        
        # Step the tracker
        active_tracks = tracker.step(meas_t)
        
        for track in active_tracks:
            # We don't want to record newborn tracks until they've proven themselves
            if track.age >= 4: 
                history = track.get_best_history()
                track_records = []
                
                # history is chronological. If we reverse it, we know the last state 
                # corresponds to the current frame 't', the one before to 't-1', etc.
                for i, state in enumerate(reversed(history)):
                    track_records.append({
                        'time': t - i,
                        'id': track.track_id,
                        'x': state[0],
                        'y': state[1],
                        'vx': state[2],
                        'vy': state[3]
                    })
                
                # Reverse back so it reads chronologically from start to finish
                track_records.reverse()
                
                # Overwrite the dictionary entry with this track's latest, smoothest history
                all_histories[track.track_id] = track_records
                
    # Flatten the dictionary into a single list for the DataFrame
    results = []
    for records in all_histories.values():
        results.extend(records)
            
    return pd.DataFrame(results)


# --- 1. Statistical Summary ---
def print_tomht_stats(meas_df: pd.DataFrame, tracked_df: pd.DataFrame):
    """Prints a summary of the TOMHT tracker's performance."""
    print("\n" + "="*50)
    print("             TOMHT PERFORMANCE SUMMARY")
    print("="*50)
    
    # Basic Counts
    total_frames = meas_df['time'].nunique()
    total_detections = len(meas_df)
    total_track_points = len(tracked_df)
    unique_tracks = tracked_df['id'].nunique() if not tracked_df.empty else 0
    
    print(f"Frames Processed      : {total_frames}")
    print(f"Raw Detections        : {total_detections}")
    print(f"Total Tracked Points  : {total_track_points}")
    print(f"Unique Tracks Formed  : {unique_tracks}")
    
    print("-" * 50)
    print(" TRACK LIFESPAN METRICS")
    print("-" * 50)
    
    if unique_tracks > 0:
        track_lengths = tracked_df.groupby('id').size()
        print(f"Max Track Length      : {track_lengths.max()} frames")
        print(f"Average Track Length  : {track_lengths.mean():.1f} frames")
        
        # Categorize tracks by length
        short_tracks = (track_lengths <= 5).sum()
        med_tracks = ((track_lengths > 5) & (track_lengths <= 20)).sum()
        long_tracks = (track_lengths > 20).sum()
        
        print(f"Short Tracks (<=5)    : {short_tracks}")
        print(f"Medium Tracks (6-20)  : {med_tracks}")
        print(f"Long Tracks (>20)     : {long_tracks}")
    else:
        print("No tracks were formed!")
        
    print("="*50 + "\n")


# --- 2. Static Plotting ---
def plot_tomht_static(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str, top_n: int = 15):
    """Plots all raw detections and overlays the tracker's longest continuous paths."""
    plt.figure(figsize=(12, 8))
    
    # 1. Scatter all raw detections in the background
    plt.scatter(meas_df['x'], meas_df['y'], c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # 2. Draw the tracks
    if not tracked_df.empty:
        # Calculate the length of each track and sort descending
        track_lengths = tracked_df.groupby('id').size().sort_values(ascending=False)
        
        # Get the IDs of the top N longest tracks
        top_track_ids = track_lengths.head(top_n).index.tolist()
        
        # Plot only these specific tracks
        for track_id in top_track_ids:
            grp = tracked_df[tracked_df['id'] == track_id]
            plt.plot(
                grp['x'], 
                grp['y'], 
                marker='.', 
                markersize=4, 
                linewidth=2.0, 
                label=f'Track {track_id} (len: {len(grp)})'
            )
            
    plt.title(f"TOMHT Output vs. Raw Detections (Top {top_n} Longest Tracks)")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    # Draw legend 
    if not tracked_df.empty:
        # We no longer need to truncate handles because we only plotted top_n tracks
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved static plot to {output_path}")


# --- 3. Animation ---
def animate_tomht(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str):
    """Creates a GIF showing raw detections per frame and active tracks."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set dynamic limits based on the data
    all_x = pd.concat([meas_df['x'], tracked_df['x']])
    all_y = pd.concat([meas_df['y'], tracked_df['y']])
    
    ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
    ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    time_steps = sorted(meas_df['time'].unique())
    colors = plt.cm.tab20.colors 
    
    # Persistent objects
    raw_scatter = ax.scatter([], [], c='gray', s=20, alpha=0.5, label='Current Detections')
    lines = {}
    
    def update(frame_time):
        ax.set_title(f"TOMHT Tracking - Frame {frame_time}")
        
        # 1. Update raw detections for this specific frame
        current_meas = meas_df[meas_df['time'] == frame_time]
        if not current_meas.empty:
            raw_scatter.set_offsets(current_meas[['x', 'y']].to_numpy())
        else:
            raw_scatter.set_offsets(np.empty((0, 2)))
            
        # 2. Update track histories up to this frame
        current_tracks = tracked_df[tracked_df['time'] <= frame_time]
        active_ids_this_frame = tracked_df[tracked_df['time'] == frame_time]['id'].unique()
        
        for track_id in current_tracks['id'].unique():
            trk_history = current_tracks[current_tracks['id'] == track_id]
            
            # Create a line if it doesn't exist
            if track_id not in lines:
                color = colors[int(track_id) % len(colors)]
                line, = ax.plot([], [], '-', color=color, linewidth=2)
                lines[track_id] = line
                
            # Update the line data
            lines[track_id].set_data(trk_history['x'], trk_history['y'])
            
            # Fade out tracks that are no longer active in the current frame
            if track_id not in active_ids_this_frame:
                lines[track_id].set_alpha(0.3)
            else:
                lines[track_id].set_alpha(1.0)
                
        return [raw_scatter] + list(lines.values())

    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=200, blit=False
    )
    
    ani.save(output_path, writer='pillow')
    plt.close()
    print(f"Saved animation to {output_path}")

def rank_tracks_by_likelihood(tracked_df: pd.DataFrame, noise_std: float = 2.0, max_allowed_rmse: float = 3.0) -> pd.DataFrame:
    """
    Retroactively scores tracks based on length AND kinematic fit.
    """
    if tracked_df.empty:
        return pd.DataFrame()

    scores = []
    
    for track_id, group in tracked_df.groupby('id'):
        n_points = len(group)
        
        if n_points < 3:
            continue
            
        t = group['time'].to_numpy()
        x = group['x'].to_numpy()
        y = group['y'].to_numpy()
        
        # Fit the line and get residuals
        p_x = np.polyfit(t, x, 1)
        p_y = np.polyfit(t, y, 1)
        
        x_fit = np.polyval(p_x, t)
        y_fit = np.polyval(p_y, t)
        
        sq_residuals_x = (x - x_fit)**2
        sq_residuals_y = (y - y_fit)**2
        sum_sq_residuals = np.sum(sq_residuals_x + sq_residuals_y)
        
        # Calculate standard metrics
        term1 = -n_points * np.log(2 * np.pi * (noise_std**2))
        term2 = - (1.0 / (2 * (noise_std**2))) * sum_sq_residuals
        log_likelihood = term1 + term2
        mean_ll = log_likelihood / n_points
        rmse = np.sqrt(sum_sq_residuals / n_points)
        
        # --- NEW: COMPOSITE SCORE ---
        # Formula: Length * (Max_Error - Actual_Error)
        # This heavily rewards long tracks, as long as they stay under the max error threshold.
        # If RMSE is worse than max_allowed_rmse, the multiplier becomes 0 (or negative).
        accuracy_multiplier = max(0.0, max_allowed_rmse - rmse)
        composite_score = n_points * accuracy_multiplier
        
        scores.append({
            'track_id': track_id,
            'length': n_points,
            'composite_score': composite_score, # Our new ranking metric
            'rmse_pixels': rmse,
            'mean_log_likelihood': mean_ll
        })

    scores_df = pd.DataFrame(scores)
    
    if not scores_df.empty:
        # Sort by the new composite score (highest is best)
        scores_df = scores_df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
        
    return scores_df

def plot_ranked_tracks(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, scores_df: pd.DataFrame, output_path: str, top_n: int = 10):
    """
    Plots the raw detections and overlays only the top N most likely tracks 
    based on their kinematic log-likelihood scores.
    """
    if tracked_df.empty or scores_df.empty:
        print("No tracks or scores to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    # 1. Scatter all raw detections in the background
    plt.scatter(meas_df['x'], meas_df['y'], c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # 2. Extract the IDs of the top N tracks
    top_track_ids = scores_df.head(top_n)['track_id'].tolist()
    
    # Use a nice color map for the top tracks
    colors = plt.cm.tab10.colors
    
    # 3. Draw the best tracks
    for rank, track_id in enumerate(top_track_ids):
        # Get the track's data
        grp = tracked_df[tracked_df['id'] == track_id]
        
        # Get the track's score stats for the legend
        stats = scores_df[scores_df['track_id'] == track_id].iloc[0]
        rmse = stats['rmse_pixels']
        
        color = colors[rank % len(colors)]
        
        # Plot the track line and points
        plt.plot(
            grp['x'], 
            grp['y'], 
            marker='o', 
            markersize=5, 
            linewidth=2.5, 
            color=color,
            label=f'Rank {rank + 1}: ID {track_id} (RMSE: {rmse:.2f} px)'
        )
        
        # Optional: Add a text label right next to the start of the track
        start_x, start_y = grp.iloc[0]['x'], grp.iloc[0]['y']
        plt.text(start_x + 5, start_y + 5, f"#{rank + 1}", color=color, fontweight='bold')

    plt.title(f"TOMHT Output: Top {top_n} Most Likely Kinematic Tracks")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    # Place legend outside the plot area so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved ranked tracks plot to {output_path}")   


# --- 4. Main Execution Block ---
if __name__ == "__main__":
    csv_path = "results/pipeline_output/master_detections_with_covariance.csv"
    output_dir = "results/tomht_eval/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load raw data for plotting
    try:
        raw_df = pd.read_csv(csv_path)
        raw_df = raw_df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        exit()

    print("Running TOMHT over pipeline data...")
    tracked_df = process_pipeline_data(csv_path)
    
    if tracked_df.empty:
        print("Warning: Tracker returned no tracks. Check your gate_distance or Kalman noise parameters.")
    else:
        # Generate Outputs
        print_tomht_stats(raw_df, tracked_df)
        plot_tomht_static(raw_df, tracked_df, os.path.join(output_dir, "tomht_static.png"))
        print("Generating animation (this may take a minute)...")
        animate_tomht(raw_df, tracked_df, os.path.join(output_dir, "tomht_animation.gif"))
        print("All done! Check the results folder.")

    print("Running TOMHT over pipeline data...")
    tracked_df = process_pipeline_data(csv_path)
    
    if tracked_df.empty:
        print("Warning: Tracker returned no tracks.")
    else:
        # 1. Print Standard Stats
        print_tomht_stats(raw_df, tracked_df)
        
        # 2. Score and Rank the Tracks
        print("Scoring tracks based on kinematic likelihood...")
        scores_df = rank_tracks_by_likelihood(tracked_df, noise_std=2.0)
        
        print("\n--- TOP 10 TRACKS ---")
        print(scores_df.head(10).to_string(index=False))
        
        # 3. Generate the Plots
        print("\nGenerating visual outputs...")
        plot_tomht_static(raw_df, tracked_df, os.path.join(output_dir, "tomht_static_all.png"))
        
        # --- NEW CALL ---
        plot_ranked_tracks(raw_df, tracked_df, scores_df, os.path.join(output_dir, "tomht_ranked_top10.png"), top_n=10)
        
        # animate_tomht(raw_df, tracked_df, os.path.join(output_dir, "tomht_animation.gif"))
        print("All done! Check the results folder.")