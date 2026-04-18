import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_ekf_diagnostics(csv_path="results/tomht_eval/ekf_updates_diagnostics.csv", output_dir="results/tomht_eval/"):
    # 1. Load the data
    if not os.path.exists(csv_path):
        print(f"[!] Could not find {csv_path}. Make sure the tracker ran successfully.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Optional: Filter to only show tracks that lived for at least N frames to reduce noise
    track_counts = df['track_id'].value_counts()
    valid_tracks = track_counts[track_counts > 5].index
    df_filtered = df[df['track_id'].isin(valid_tracks)]
    
    print(f"[*] Plotting diagnostics for {len(valid_tracks)} tracks...")

    # ==========================================
    # PLOT 1: RESIDUALS OVER TIME (u, v, rho)
    # ==========================================
    fig_res, axs_res = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_res.suptitle("EKF Measurement Residuals (Innovation) Over Time", fontsize=16)
    
    for track_id, group in df_filtered.groupby('track_id'):
        axs_res[0].plot(group['time'], group['res_u'], marker='.', linestyle='-', alpha=0.6, label=f'Track {track_id}')
        axs_res[1].plot(group['time'], group['res_v'], marker='.', linestyle='-', alpha=0.6)
        axs_res[2].plot(group['time'], group['res_rho'], marker='.', linestyle='-', alpha=0.6)

    # u-residual
    axs_res[0].set_ylabel("Residual U (m)")
    axs_res[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs_res[0].grid(True, linestyle=':', alpha=0.7)
    
    # v-residual
    axs_res[1].set_ylabel("Residual V (m)")
    axs_res[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs_res[1].grid(True, linestyle=':', alpha=0.7)
    
    # rho-residual
    axs_res[2].set_ylabel("Residual Rho (m)")
    axs_res[2].set_xlabel("Time (Frames)")
    axs_res[2].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs_res[2].grid(True, linestyle=':', alpha=0.7)
    
    # Only show legend on the top plot (and limit it if too many tracks)
    if len(valid_tracks) <= 10:
        axs_res[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
    plt.tight_layout()
    res_save_path = os.path.join(output_dir, "ekf_residuals_plot.png")
    fig_res.savefig(res_save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved Residuals plot to {res_save_path}")

    # ==========================================
    # PLOT 2: VARIANCES OVER TIME (x, y, z)
    # ==========================================
    fig_var, axs_var = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig_var.suptitle("EKF State Variance Over Time", fontsize=16)
    
    for track_id, group in df_filtered.groupby('track_id'):
        axs_var[0].plot(group['time'], group['var_x'], marker='.', linestyle='-', alpha=0.6, label=f'Track {track_id}')
        axs_var[1].plot(group['time'], group['var_y'], marker='.', linestyle='-', alpha=0.6)
        axs_var[2].plot(group['time'], group['var_z'], marker='.', linestyle='-', alpha=0.6)

    # X-variance
    axs_var[0].set_ylabel("Variance X ($m^2$)")
    axs_var[0].set_yscale('log') # Log scale is usually best for variance collapsing
    axs_var[0].grid(True, which="both", linestyle=':', alpha=0.7)
    
    # Y-variance
    axs_var[1].set_ylabel("Variance Y ($m^2$)")
    axs_var[1].set_yscale('log')
    axs_var[1].grid(True, which="both", linestyle=':', alpha=0.7)
    
    # Z-variance
    axs_var[2].set_ylabel("Variance Z ($m^2$)")
    axs_var[2].set_xlabel("Time (Frames)")
    axs_var[2].set_yscale('log')
    axs_var[2].grid(True, which="both", linestyle=':', alpha=0.7)
    
    if len(valid_tracks) <= 10:
        axs_var[0].legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()
    var_save_path = os.path.join(output_dir, "ekf_variances_plot.png")
    fig_var.savefig(var_save_path, dpi=300, bbox_inches='tight')
    print(f"[✓] Saved Variances plot to {var_save_path}")
    
    plt.show() # Uncomment if you want them to pop up on screen

if __name__ == "__main__":
    plot_ekf_diagnostics()