import os
import pandas as pd
from typing import cast

# Simulation Imports
from src.simulate_fits_data import simulate_fits_data

# Detection Pipeline Imports
from src.process_fits_files import process_fits_directory 

# Tracking Imports
from utils.config_loader import TrackerConfig
from src.tomht import TOMHTTracker
from utils.post_process import (
    plot_longest_tracks,
    animate_tracks,
    print_tomht_stats
)

def simulate_data(trajectory_csv: str):
    """Step 1: Generates synthetic FITS images from a trajectory CSV."""
    print("\n" + "="*40)
    print(" STEP 1: SIMULATING IMAGE DATA")
    print("="*40)
    
    # Call data simulation function
    simulate_fits_data(trajectory_csv,verbose=True)


def detect_sources(INPUT_FITS_DIR: str, OUTPUT_CSV_DIR: str, OUTPUT_PLOT_DIR: str):
    """Step 2: Runs background subtraction and matched filtering on the simulated FITS files."""
    print("\n" + "="*40)
    print(" STEP 2: SOURCE DETECTION (MATCHED FILTER)")
    print("="*40)

    
    process_fits_directory(
        input_dir=INPUT_FITS_DIR, 
        output_csv_dir=OUTPUT_CSV_DIR, 
        output_plot_dir=OUTPUT_PLOT_DIR,
        sigma_psf=2.0, 
        threshold_factor=5.0,
        skip_bg_sub=False,
        verbose=True
    )
    

def run_tracker(csv_path: str, verbose: bool = True):
    """
    Step 3: Runs the TOMHT Tracker over the detections generated in Step 2.
    """
    print("\n" + "="*40)
    print(" STEP 3: TOMHT MULTI-TARGET TRACKING")
    print("="*40)
    
    output_dir = "results/tomht_eval/"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        raw_df = pd.read_csv(csv_path)
        raw_df = raw_df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})
    except FileNotFoundError:
        print(f"[!] Error: Could not find {csv_path}. Did Step 2 fail?")
        return

    if verbose: print(f"[*] Running TOMHT over {len(raw_df)} pipeline detections...")
    
    config = TrackerConfig.from_jsonx("config/parameters.jsonx")
    tracker = TOMHTTracker(config)
    n_scan = config.n_scan_window 
    results = []
    
    # Tracking Loop
    unique_times = raw_df['time'].unique()
    for i, (t_raw, group) in enumerate(raw_df.groupby('time')):
        t = cast(int, t_raw) 
        meas_t = group[['x', 'y']].to_numpy()
        active_tracks = tracker.step(meas_t)

        for track in active_tracks:
            if len(track.best_hypothesis.history_states) >= n_scan: 
                delayed_state = track.best_hypothesis.history_states[-n_scan]
                results.append({
                    'time': t - n_scan,
                    'id': track.track_id,
                    'x': delayed_state[0], 'y': delayed_state[1],
                    'vx': delayed_state[2], 'vy': delayed_state[3]
                })
        
        if not verbose:
            print(f"\r[+] Tracking Progress: {i+1}/{len(unique_times)} frames", end='', flush=True)

    if not verbose: print()

    tracked_df = pd.DataFrame(results)
    
    if tracked_df.empty:
        print("[!] Warning: Tracker returned no tracks. Check gating or noise parameters.")
    else:
        if verbose:
            print_tomht_stats(raw_df, tracked_df)
            print("[*] Generating visualization artifacts...")
        
        plot_longest_tracks(raw_df, tracked_df, os.path.join(output_dir, "tomht_longest_tracks.png"))
        animate_tracks(raw_df, tracked_df, os.path.join(output_dir, "tomht_animation.gif"))
        
        if verbose: print(f"[✓] Tracker visualizations saved to: {output_dir}")

    return tracked_df


if __name__ == "__main__":
    # --- PIPELINE CONFIGURATION ---
    TARGET_TRAJECTORY_CSV = r"data\cso_data\Slowing_one.csv" ## <-- UPDATE WITH TRAJECTORY CSV PATH
    DETECTIONS_CSV_OUTPUT = "results/detections/master_detections_with_covariance.csv"

    # Ensure the master results directory exists
    os.makedirs("results", exist_ok=True)
    
    # --- EXECUTE PIPELINE ---
    simulate_data(trajectory_csv=TARGET_TRAJECTORY_CSV)
    
    detect_sources(
        INPUT_FITS_DIR="results/simulated_data/fits",
        OUTPUT_CSV_DIR="results/detections",
        OUTPUT_PLOT_DIR="results/detection_frames"
    )
    
    run_tracker(csv_path=DETECTIONS_CSV_OUTPUT)
    
    print("\n========================================")
    print(" END-TO-END TOMHT TRACKING COMPLETE")
    print("========================================")