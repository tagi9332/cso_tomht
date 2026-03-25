import numpy as np
import pandas as pd
import os

# Local imports
from utils.photometrics.calculate_flux_for_snr import calculate_flux_for_snr
from utils.photometrics.calculate_optical_properties import calculate_optical_properties

def add_gaussian_source(image, x, y, flux, sigma, roi_sigma_mult=5):
    """
    Adds a 2D Gaussian source to the image array in place using a Region of Interest (ROI).
    """
    h, w = image.shape
    roi_radius = int(roi_sigma_mult * sigma)
    
    # integer bounds
    x_min = max(0, int(x - roi_radius))
    x_max = min(w, int(x + roi_radius + 1))
    y_min = max(0, int(y - roi_radius))
    y_max = min(h, int(y + roi_radius + 1))
    
    if x_min >= x_max or y_min >= y_max:
        return # Source is off-image
        
    # Grid generation only for the ROI
    X_roi, Y_roi = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    
    # 2D Gaussian
    normalization = flux / (2 * np.pi * sigma**2)
    dist_sq = (X_roi - x)**2 + (Y_roi - y)**2
    psf_values = normalization * np.exp(-dist_sq / (2 * sigma**2))
    
    image[y_min:y_max, x_min:x_max] += psf_values

def apply_noise(clean_image, read_noise_std):
    """
    Applies Shot noise (Poisson) and Read noise (Gaussian).
    """
    # Poisson (Shot) Noise - requires non-negative input
    # Clip negative values just in case floats drifted below 0
    clean_image = np.maximum(clean_image, 0)
    image_with_shot = np.random.poisson(clean_image)
    
    # Gaussian (Read) Noise
    read_noise = np.random.normal(0, read_noise_std, clean_image.shape)
    
    return image_with_shot + read_noise

def run_simulation(config, trajectory_df, output_dir=None):
    """
    Main driver function driven by a trajectory CSV/DataFrame.
    Args:
        config (dict): Dictionary containing sensor/optical parameters.
        trajectory_df (pd.DataFrame): DataFrame with columns ['x', 'y', 'id', 'time']
        output_dir (str): Path to save results. If None, no files are written.
    Returns:
        tuple: (list_of_frames, ground_truth_dataframe)
    """
    
    # Setup Optical Parameters
    sigma_psf, r_airy = calculate_optical_properties(
        config['wavelength'], config['f_len'], config['D'], config['pixel_pitch']
    )
    
    # Window setup for SNR calculations
    raw_width = 6 * sigma_psf
    window_width = int(np.ceil(raw_width))
    if window_width % 2 == 0: window_width += 1
    n_pix_window = window_width ** 2
    
    # Map Unique Object IDs to Fluxes based on config SNRs
    unique_ids = sorted(trajectory_df['id'].unique())
    id_to_flux = {}
    id_to_snr = {}
    
    for i, obj_id in enumerate(unique_ids):
        # Assign SNR from config list (loop back if there are more objects than SNRs)
        snr = config['snr_targets'][i % len(config['snr_targets'])]
        flux = calculate_flux_for_snr(snr, n_pix_window, config['background_mean'], config['read_noise_std'])
        id_to_flux[obj_id] = flux
        id_to_snr[obj_id] = snr

    # Extract unique times to represent frames
    unique_times = sorted(trajectory_df['time'].unique())
    print(f"Starting simulation: {len(unique_times)} frames, {len(unique_ids)} unique objects.")

    ground_truth_records = []
    all_frames_data = []
    
    # Simulation Loop
    for frame_idx, current_time in enumerate(unique_times):
        
        # Initialize Background
        clean_signal = np.zeros((config['img_size'], config['img_size'])) + config['background_mean']
        
        # Get all objects active at this specific time/frame
        active_objects = trajectory_df[trajectory_df['time'] == current_time]
        
        for _, row in active_objects.iterrows():
            true_x, true_y = row['x'], row['y']
            obj_id = row['id']
            flux = id_to_flux[obj_id]
            
            # Add the point spread function to the image
            add_gaussian_source(clean_signal, true_x, true_y, flux, sigma_psf)
            
            ground_truth_records.append({
                "Frame": frame_idx,
                "Time": current_time,
                "Object_ID": obj_id,
                "Target_SNR": id_to_snr[obj_id],
                "True_X": true_x,
                "True_Y": true_y,
                "Signal_Flux": flux
            })
            
        # Apply Noise
        final_frame = apply_noise(clean_signal, config['read_noise_std'])
        all_frames_data.append(final_frame)

    # Wrap up results
    df_truth = pd.DataFrame(ground_truth_records)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_truth.to_csv(os.path.join(output_dir, "ground_truth.csv"), index=False)
        print(f"Results saved to {output_dir}")

    return all_frames_data, df_truth