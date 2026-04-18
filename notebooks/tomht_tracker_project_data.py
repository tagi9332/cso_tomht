import os,sys
import pandas as pd
from typing import cast

# Set project root and add to sys.path for local imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Local imports from the project
from src.track import Track, Hypothesis
from utils.config_loader import TrackerConfig
from utils.kalman_filter import KalmanFilter2D
from src.tomht import TOMHTTracker 
from src.tomht import TOMHTTracker
from utils.post_process import print_tomht_stats, plot_longest_tracks, animate_tracks 


def run_tracker(csv_path: str, generate_gif: bool = False, verbose: bool = True):
    """Step 3: Runs the TOMHT Tracker over the detections generated in Step 2."""
    print("\n" + "="*40)
    print(" STEP 3: TOMHT MULTI-TARGET TRACKING")
    print("="*40)
    
    output_dir = "results/tomht_eval/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prep the data
    try:
        raw_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Error: Could not find the file at {csv_path}")
        print("Please ensure Step 2 completed successfully and the path is correct.")
        return pd.DataFrame()

    raw_df = raw_df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})

    if verbose: 
        print(f"[*] Running TOMHT over {len(raw_df)} pipeline detections...")
    
    # Initialize Tracker
    config = TrackerConfig.from_jsonx("config/parameters.jsonx")
    tracker = TOMHTTracker(config)
    n_scan = config.n_scan_window 
    results = []
    
    # Tracking Loop
    unique_times = raw_df['time'].unique()
    for i, (t_raw, group) in enumerate(raw_df.groupby('time')):
        t = cast(int, t_raw) 
        meas_t = group[['x', 'y']].to_numpy()
        
        # Step the filter
        active_tracks = tracker.step(meas_t)

        # Extract delayed states based on N-scan pruning window
        for track in active_tracks:
            if len(track.best_hypothesis.history_states) >= n_scan: 
                delayed_state = track.best_hypothesis.history_states[-n_scan]
                results.append({
                    'time': t - n_scan,
                    'id': track.track_id,
                    'x': delayed_state[0], 'y': delayed_state[1],
                    'vx': delayed_state[2], 'vy': delayed_state[3]
                })
        
        # Progress indicator
        if not verbose:
            print(f"\r[+] Tracking Progress: {i+1}/{len(unique_times)} frames", end='', flush=True)

    if not verbose: 
        print() # Newline after progress bar finishes

    tracked_df = pd.DataFrame(results)
    
    # Evaluation & Visualization
    if tracked_df.empty:
        print("[!] Warning: Tracker returned no tracks. Check gating or noise parameters.")
    else:
        if verbose:
            print_tomht_stats(raw_df, tracked_df)
            print("[*] Generating visualization artifacts...")
        
        plot_longest_tracks(raw_df, tracked_df, os.path.join(output_dir, "tomht_longest_tracks.png"), display_plots=verbose, n_tracks=1000)
        
        if generate_gif:
            animate_tracks(raw_df, tracked_df, os.path.join(output_dir, "tomht_animation.gif"))
        
        if verbose: 
            print(f"[✓] Tracker visualizations saved to: {output_dir}")

    return tracked_df


# =====================================================================
# MAIN EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    # Path to detection file
    DETECTIONS_CSV_OUTPUT = os.path.join("results", "pipeline_output", "master_detections_with_covariance.csv")
    
    # Execute the tracker and get the tracked DataFrame
    tracked_dataframe = run_tracker(
        csv_path=DETECTIONS_CSV_OUTPUT, 
        generate_gif=False, 
        verbose=True
    )

    print("\n========================================")
    print(" END-TO-END TOMHT TRACKING COMPLETE")
    print("========================================")