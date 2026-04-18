import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_gating_diagnostics(csv_path=r"results/tomht_eval/ekf_gating_diagnostics.csv"):
    if not os.path.exists(csv_path):
        print(f"[!] File not found: {csv_path}")
        return

    # 1. Load the data
    df = pd.read_csv(csv_path)

    # 2. Parse the z_true numpy array strings
    def parse_z_true(z_str):
        # Remove brackets and split by any whitespace
        clean_str = z_str.replace('[', '').replace(']', '').strip()
        vals = [float(v) for v in clean_str.split()]
        return vals[0], vals[1], vals[2]

    # Expand the parsed values into separate columns
    df[['u', 'v', 'rho']] = df['z_true'].apply(lambda x: pd.Series(parse_z_true(x)))

    # 3. Create the plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    fig.suptitle("EKF Gating Diagnostics Analysis", fontsize=16)

    # We use scatter plots (linestyle='') because there are multiple hypotheses/measurements per time step
    for track_id, group in df.groupby('track_id'):
        axs[0].plot(group['time'], group['mahalanobis_dist'], marker='o', markersize=4, linestyle='', alpha=0.5, label=f'Track {track_id}')
        axs[1].plot(group['time'], group['u']*1e6, marker='o', markersize=4, linestyle='', alpha=0.5)
        axs[2].plot(group['time'], group['v']*1e6, marker='o', markersize=4, linestyle='', alpha=0.5)
        axs[3].plot(group['time'], group['rho']*1e6, marker='o', markersize=4, linestyle='', alpha=0.5)

    # Plot 1: Mahalanobis Distance
    axs[0].set_ylabel("Mahalanobis Dist")
    axs[0].set_yscale('log') # Log scale helps see variations in tiny distances
    axs[0].grid(True, linestyle='--', alpha=0.7)
    axs[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

    # Plot 2: U measurement
    axs[1].set_ylabel("u (meters)")
    axs[1].grid(True, linestyle='--', alpha=0.7)

    # Plot 3: V measurement
    axs[2].set_ylabel("v (meters)")
    axs[2].grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Rho measurement (Range)
    axs[3].set_ylabel("rho (meters)")
    axs[3].set_xlabel("Time (Frames)")
    axs[3].grid(True, linestyle='--', alpha=0.7)
    
    # Add a red line at 0 for the range plot
    axs[3].axhline(0, color='red', linestyle='-', linewidth=2, alpha=0.5, label='Zero Range (Camera Lens)')
    axs[3].legend(loc='upper right')

    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(os.path.dirname(csv_path), "gating_analysis_plot.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"[*] Saved gating analysis plot to: {save_path}")
    
    plt.show()

# Run the function
plot_gating_diagnostics()