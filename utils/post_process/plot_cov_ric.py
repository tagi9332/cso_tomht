import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_covariance_evolution_ric(cov_df: pd.DataFrame, output_path: str = "results/tomht_eval/covariance_ric_plot.png"):
    """Plots the 6D state standard deviations over time broken out into the RIC frame."""
    if cov_df.empty:
        print("[!] No covariance data to plot.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a 2x3 grid: Rows = [Position, Velocity], Cols = [Radial, In-track, Cross-track]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    
    # Group by track ID so we can plot lines for each individual target
    for track_id, track_data in cov_df.groupby('id'):
        times = track_data['time']
        
        # --- Row 0: Position Standard Deviations (m) ---
        axes[0, 0].plot(times, track_data['std_x'], label=f'Track {track_id}')
        axes[0, 1].plot(times, track_data['std_y'], label=f'Track {track_id}')
        axes[0, 2].plot(times, track_data['std_z'], label=f'Track {track_id}')
        
        # --- Row 1: Velocity Standard Deviations (m/s) ---
        axes[1, 0].plot(times, track_data['std_vx'], label=f'Track {track_id}', linestyle='--')
        axes[1, 1].plot(times, track_data['std_vy'], label=f'Track {track_id}', linestyle='--')
        axes[1, 2].plot(times, track_data['std_vz'], label=f'Track {track_id}', linestyle='--')

    # Formatting Titles
    axes[0, 0].set_title("Radial Position (X)", fontsize=13, fontweight='bold')
    axes[0, 1].set_title("In-track Position (Y / Range)", fontsize=13, fontweight='bold')
    axes[0, 2].set_title("Cross-track Position (Z)", fontsize=13, fontweight='bold')
    
    axes[1, 0].set_title("Radial Velocity (Vx)", fontsize=13, fontweight='bold')
    axes[1, 1].set_title("In-track Velocity (Vy)", fontsize=13, fontweight='bold')
    axes[1, 2].set_title("Cross-track Velocity (Vz)", fontsize=13, fontweight='bold')

    # Formatting Y-Axis Labels & Scales
    for j in range(3):
        axes[0, j].set_ylabel("Std Dev (meters)")
        axes[1, j].set_ylabel("Std Dev (m/s)")
        axes[1, j].set_xlabel("Time (Frame Index)")
        
        for i in range(2):
            axes[i, j].grid(True, linestyle=':', alpha=0.7)
            axes[i, j].set_yscale('log')

    # Add a single master legend outside the subplots

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"[*] RIC Covariance plot saved to {output_path}")