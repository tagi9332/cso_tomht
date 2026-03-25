import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from typing import List, Any

from utils.photometrics.calculate_optical_properties import calculate_optical_properties

def plot_summary_frame(
    frames: List[Any], 
    truth_df: pd.DataFrame, 
    sim_config: dict, 
    output_dir: str, 
    vmin: float, 
    vmax: float
) -> None:
    """Generates and saves a static plot of the final frame with overlaid truth trajectories."""
    
    # Calculate window size for plotting circles
    sigma_psf, _ = calculate_optical_properties(
        sim_config['wavelength'], sim_config['f_len'], 
        sim_config['D'], sim_config['pixel_pitch']
    )
    window_width = int(np.ceil(2 * (3 * sigma_psf)))

    fig, ax = plt.subplots(figsize=(8, 8))
    final_image = frames[-1] 

    im = ax.imshow(final_image, cmap='gray', origin='lower', interpolation='nearest', 
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Counts')
    ax.set_title(f"Synthetic Final Frame\nTotal Frames: {len(frames)}")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    # Annotate object tracks based on the truth DataFrame
    for obj_idx in truth_df['Object_ID'].unique():
        obj_data = truth_df[truth_df['Object_ID'] == obj_idx]
        
        track_x = obj_data['True_X'].values
        track_y = obj_data['True_Y'].values
        snr = obj_data.iloc[0]['Target_SNR']
        
        ax.plot(track_x, track_y, color='red', linestyle='--', alpha=0.6)
        
        end_x, end_y = track_x[-1], track_y[-1]
        circ = Circle((end_x, end_y), radius=window_width/2, edgecolor='red', facecolor='none', lw=1.5)
        ax.add_patch(circ)
        ax.text(end_x, end_y + 3, f"ID {obj_idx}\nSNR {snr}", color='red', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "simulation_final_frame.png"), dpi=100)
    plt.close(fig) # Always close plots in utilities to free memory


def create_simulation_gif(
    frames: List[Any], 
    output_dir: str, 
    vmin: float, 
    vmax: float
) -> None:
    """Generates an animated GIF of the simulated frames."""

    fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
    ax_anim.set_title("Simulation Animation")
    ax_anim.set_xlabel("Pixel X")
    ax_anim.set_ylabel("Pixel Y")

    ims = []
    for frame in frames:
        im = ax_anim.imshow(frame, cmap='gray', origin='lower', animated=True,
                            vmin=vmin, vmax=vmax)
        ims.append([im])

    ani = animation.ArtistAnimation(fig_anim, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save(os.path.join(output_dir, "simulation_animation.gif"), writer='pillow', fps=10)
    plt.close(fig_anim)
