import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation
from astropy.io import fits

from utils.obj_sim.image_sim import run_simulation
from utils.photometrics.calculate_optical_properties import calculate_optical_properties

# 1. CONFIGURATION
sim_config = {
    'f_len': 4.0,
    'D': 0.5,
    'wavelength': 500e-9,
    'pixel_pitch': 1.5e-6,
    'read_noise_std': 10.0,
    'background_mean': 100.0,
    'img_size': 100,
    'snr_targets': [10,3] # Assigned to objects in the CSV sequentially
}

# 2. LOAD TRAJECTORY DATA
trajectory_df = pd.read_csv(r"data\Crossing.csv")

# 3. RUN SIMULATION 
# (Changed "/results" to "results" to avoid attempting to write to your hard drive's root directory)
frames, truth_df = run_simulation(sim_config, trajectory_df, output_dir="results")

# ---------------- NEW: INDIVIDUAL FRAME EXPORT ----------------
print("Exporting individual frames to FITS and PNG...")

fits_dir = "results/fits"
png_dir = "results/png"
os.makedirs(fits_dir, exist_ok=True)
os.makedirs(png_dir, exist_ok=True)

# Calculate the same contrast limits used in the plots
vmin_limit = sim_config['background_mean'] - 3*sim_config['read_noise_std']
vmax_limit = sim_config['background_mean'] + 500

for i, frame in enumerate(frames):
    # Save as FITS (raw float/int data)
    hdu = fits.PrimaryHDU(frame)
    fits_filename = os.path.join(fits_dir, f"frame_{i:04d}.fits")
    hdu.writeto(fits_filename, overwrite=True)
    
    # Save as PNG (scaled grayscale image for easy viewing)
    png_filename = os.path.join(png_dir, f"frame_{i:04d}.png")
    plt.imsave(png_filename, frame, cmap='gray', vmin=vmin_limit, vmax=vmax_limit)

print(f"Exported {len(frames)} frames.")
# --------------------------------------------------------------

# Re-calculate window size for plotting circles
sigma_psf, r_airy = calculate_optical_properties(
    sim_config['wavelength'], sim_config['f_len'], sim_config['D'], sim_config['pixel_pitch']
)
window_width = int(np.ceil(2 * (3 * sigma_psf)))

# 4. OUTPUT & VISUALIZATION 
print("Generating plots...")

fig, ax = plt.subplots(figsize=(8, 8))
final_image = frames[-1] 

im = ax.imshow(final_image, cmap='gray', origin='lower', interpolation='nearest', 
               vmin=vmin_limit, vmax=vmax_limit)
plt.colorbar(im, ax=ax, label='Counts')
ax.set_title(f"Synthetic Final Frame\nTotal Frames: {len(frames)}")
ax.set_xlabel("Pixel X")
ax.set_ylabel("Pixel Y")

# Annotate object tracks based on the truth DataFrame
for obj_idx in truth_df['Object_ID'].unique():
    obj_data = truth_df[truth_df['Object_ID'] == obj_idx]
    
    # Get all X, Y coords for the track line
    track_x = obj_data['True_X'].values
    track_y = obj_data['True_Y'].values
    snr = obj_data.iloc[0]['Target_SNR']
    
    # Plot the full trajectory line
    ax.plot(track_x, track_y, color='red', linestyle='--', alpha=0.6)
    
    # Put a circle at its FINAL location in this frame
    end_x, end_y = track_x[-1], track_y[-1]
    circ = Circle((end_x, end_y), radius=window_width/2, edgecolor='red', facecolor='none', lw=1.5)
    ax.add_patch(circ)
    ax.text(end_x, end_y + 3, f"ID {obj_idx}\nSNR {snr}", color='red', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("results/simulation_final_frame.png", dpi=100)
print("Saved static plot.")

# SAVE ANIMATED GIF 
print("Generating GIF animation...")

fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_title("Simulation Animation")
ax_anim.set_xlabel("Pixel X")
ax_anim.set_ylabel("Pixel Y")

ims = []
for frame in frames:
    im = ax_anim.imshow(frame, cmap='gray', origin='lower', animated=True,
                        vmin=vmin_limit, vmax=vmax_limit)
    ims.append([im])

ani = animation.ArtistAnimation(fig_anim, ims, interval=100, blit=True, repeat_delay=1000)
ani.save("results/simulation_animation.gif", writer='pillow', fps=10)
print("Saved animation.")