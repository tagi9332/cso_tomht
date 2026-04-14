import pandas as pd
import numpy as np
import os
import json
import re
from utils.img_sim import (
    run_simulation,
    export_frames,
    plot_summary_frame,
    create_simulation_gif,
    plot_frame_grid
)

def simulate_fits_data(trajectory_csv: str, config: dict, verbose: bool = False):
    """
    Runs the image simulation pipeline for a given trajectory CSV.
    
    Args:
        trajectory_csv: Path to the input CSV.
        config: Dictionary containing simulation configuration parameters.
        verbose: If True, prints detailed status updates to the console.
    """
    if verbose:
        print(f"[*] Loading trajectory: {trajectory_csv}")
    
    # Load data first to extract the number of unique targets
    trajectory_df = pd.read_csv(trajectory_csv)

    # Determine the number of unique objects to generate appropriate SNR values
    target_col = 'Object_ID' if 'Object_ID' in trajectory_df.columns else 'id' if 'id' in trajectory_df.columns else None
    num_targets = trajectory_df[target_col].nunique() if target_col else 1

    # Sample SNRs from a Gaussian distribution
    mu, sigma = 10, 5
    sampled_snrs = np.random.normal(mu, sigma, num_targets)
    sampled_snrs = np.clip(sampled_snrs, 3.0, 10.0)
    
    # Configuration
    sim_config = {
        'f_len': config['optical_sensor']['f_len'],
        'D': config['optical_sensor']['D'],
        'wavelength': config['optical_sensor']['wavelength'],
        'pixel_pitch': config['optical_sensor']['pixel_pitch'],
        'read_noise_std': config['optical_sensor']['read_noise_std'],
        'background_mean': config['optical_sensor']['background_mean'],
        'img_size': config['optical_sensor']['img_size'],
        'snr_targets': sampled_snrs.tolist(),
        'dt': config.get('dt', 1.0)
    }

    output_dir = "results/simulated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    vmin_limit = sim_config['background_mean'] - 3 * sim_config['read_noise_std']
    vmax_limit = sim_config['background_mean'] + 500

    dt = int(sim_config['dt'])
    if verbose:
        print(f"[*] Downsampling trajectory to a {dt}-second image capture rate...")
        
    # Attempt to identify the time column and filter accordingly
    if 'time' in trajectory_df.columns:
        trajectory_df = trajectory_df[trajectory_df['time'] % dt == 0].copy()
    elif 't' in trajectory_df.columns:
        trajectory_df = trajectory_df[trajectory_df['t'] % dt == 0].copy()
    else:
        trajectory_df = trajectory_df.iloc[::dt].copy()
        
    # Reset index
    trajectory_df.reset_index(drop=True, inplace=True)

    if verbose:
        print(f"[*] Running simulation logic on {len(trajectory_df)} frames...")
    
    # Run simulation
    frames, truth_df = run_simulation(sim_config, trajectory_df, output_dir=output_dir)

    # Export frames and generate visualizations
    if verbose:
        print("[+] Simulation core complete. Generating artifacts...")
    
    export_frames(frames, output_dir, vmin_limit, vmax_limit)
    plot_summary_frame(frames, truth_df, sim_config, output_dir, vmin_limit, vmax_limit)
    create_simulation_gif(frames, output_dir, vmin_limit, vmax_limit)
    
    # Integrated grid plotting function
    plot_frame_grid(
        frames=frames, 
        n=3, 
        output_dir=output_dir, 
        vmin=vmin_limit, 
        vmax=vmax_limit
    )

    if verbose:
        print(f"[✓] Simulation successful. Outputs saved to: {output_dir}")
    
    return frames, truth_df