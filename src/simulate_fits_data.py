import pandas as pd
import os
from utils.img_sim import (
    run_simulation,
    export_frames,
    plot_summary_frame,
    create_simulation_gif
)

def simulate_fits_data(trajectory_csv: str, verbose: bool = False):
    """
    Runs the image simulation pipeline for a given trajectory CSV.
    
    Args:
        trajectory_csv: Path to the input CSV.
        verbose: If True, prints detailed status updates to the console.
    """
    if verbose:
        print(f"[*] Loading trajectory: {trajectory_csv}")
    
    # Configuration
    sim_config = {
        'f_len': 4.0,
        'D': 0.5,
        'wavelength': 500e-9,
        'pixel_pitch': 1.5e-6,
        'read_noise_std': 10.0,
        'background_mean': 100.0,
        'img_size': 100,
        'snr_targets': [10, 5] 
    }

    output_dir = "results/simulated_data"
    os.makedirs(output_dir, exist_ok=True)
    
    vmin_limit = sim_config['background_mean'] - 3 * sim_config['read_noise_std']
    vmax_limit = sim_config['background_mean'] + 500

    # Load data and run simulation
    trajectory_df = pd.read_csv(trajectory_csv)
    
    if verbose:
        print(f"[*] Running simulation logic...")
    
    # Assuming run_simulation handles its own internal verbose/logging if needed
    frames, truth_df = run_simulation(sim_config, trajectory_df, output_dir=output_dir)

    # Export frames and generate visualizations
    if verbose:
        print("[+] Simulation core complete. Generating artifacts...")
    
    export_frames(frames, output_dir, vmin_limit, vmax_limit)
    plot_summary_frame(frames, truth_df, sim_config, output_dir, vmin_limit, vmax_limit)
    create_simulation_gif(frames, output_dir, vmin_limit, vmax_limit)

    if verbose:
        print(f"[✓] Simulation successful. Outputs saved to: {output_dir}")
    
    return frames, truth_df

# Main Execution Block:
if __name__ == "__main__":
    csv_file = r"data\cso_data\Curving_toward.csv"
    
    simulate_fits_data(csv_file, verbose=True)