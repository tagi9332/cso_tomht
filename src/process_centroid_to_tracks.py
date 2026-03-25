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
        gate_distance: float = 30.0,
        max_misses: int = 3,
        max_hypotheses: int = 5
    ):
        self.kf = KalmanFilter2D(dt=dt, process_noise_std=5.0, measurement_noise_std=2.0)
        self.assoc = KDTreeAssociation(gate_distance=gate_distance, merge_clusters=True)
        
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.max_misses = max_misses
        self.max_hypotheses = max_hypotheses

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
                
                for x_pred, P_pred in predicted_hyps:
                    hyp_updates = []
                    
                    # Option 1: Missed Detection (Null Hypothesis)
                    hyp_updates.append((None, x_pred, P_pred, track.miss_log_likelihood))
                    
                    # Option 2: Associate with a gated measurement
                    for m_idx in m_idxs:
                        z = measurements[m_idx]
                        
                        # Use Mahalanobis distance to strictly gate based on covariance
                        dist = self.kf.mahalanobis_distance(x_pred, P_pred, z)
                        
                        # Only branch if the measurement is statistically plausible
                        # (A Mahalanobis distance of 3.0 captures ~99% of a 2D Gaussian)
                        if dist < 5.0: 
                            x_upd, P_upd, log_ll = self.kf.update(x_pred, P_pred, z)
                            hyp_updates.append((m_idx, x_upd, P_upd, log_ll))
                            
                    measurement_updates.append(hyp_updates)
                
                # Apply the branch-and-prune step
                track.expand_hypotheses(predicted_hyps, measurement_updates)
                track.normalise_scores()
                
                # Update track management stats based on the new best hypothesis
                if track.best_hypothesis.meas_index is None:
                    track.consecutive_misses += 1
                else:
                    track.consecutive_misses = 0

        # 4. PRUNE dead tracks
        dead_track_ids = [t_id for t_id, t in self.active_tracks.items() if t.consecutive_misses >= self.max_misses]
        for t_id in dead_track_ids:
            del self.active_tracks[t_id]
            
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
    
    tracker = TOMHTTracker(dt=1.0, gate_distance=15.0, max_misses=3)
    
    results = []
    for t in time_steps:
        # Extract [x, y] coordinates for this frame
        meas_t = df[df['time'] == t][['x', 'y']].to_numpy()
        
        # Step the tracker
        active_tracks = tracker.step(meas_t)
        
        # Record the BEST hypothesis for each active track to plot later
        for track in active_tracks:
            # We don't want to record newborn tracks until they've proven themselves (e.g., survived 2 frames)
            if track.age >= 2: 
                results.append({
                    'time': t,
                    'id': track.track_id,
                    'x': track.best_state[0],
                    'y': track.best_state[1],
                    'vx': track.best_state[2],
                    'vy': track.best_state[3]
                })
                
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
def plot_tomht_static(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str):
    """Plots all raw detections and overlays the tracker's continuous paths."""
    plt.figure(figsize=(12, 8))
    
    # 1. Scatter all raw detections in the background
    plt.scatter(meas_df['x'], meas_df['y'], c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # 2. Draw the tracks
    if not tracked_df.empty:
        # Group by track ID and plot lines
        for track_id, grp in tracked_df.groupby('id'):
            # Only plot tracks that survived a decent amount of time to reduce clutter
            if len(grp) >= 3: 
                plt.plot(grp['x'], grp['y'], marker='.', markersize=4, linewidth=1.5, label=f'Track {track_id}')
    
    plt.title("TOMHT Output vs. Raw Detections")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    # Limit legend to top 15 longest tracks to avoid crowding
    if not tracked_df.empty:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[:16], labels[:16], bbox_to_anchor=(1.05, 1), loc='upper left')
        
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