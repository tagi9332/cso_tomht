import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.cluster import DBSCAN
# import astroalign as aa

# --- CALIBRATION FRAMES ---
def create_master_frames(bias_dir, dark_dir, plot_out_dir):
    master_bias, master_dark = None, None
    os.makedirs(plot_out_dir, exist_ok=True)
    
    # Create Master Bias
    bias_files = glob.glob(os.path.join(bias_dir, "*.fit"))
    if bias_files:
        bias_stack = np.array([fits.getdata(f).astype(np.float32) for f in bias_files])
        master_bias = np.median(bias_stack, axis=0)
        print(f"[*] Created Master Bias from {len(bias_files)} frames.")
        
        # Save Bias PNG
        plt.figure(figsize=(8, 8))
        vmin, vmax = np.percentile(master_bias, 1), np.percentile(master_bias, 99)
        plt.imshow(master_bias, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='ADU (Read Noise)')
        plt.title('Master Bias Frame')
        plt.savefig(os.path.join(plot_out_dir, "master_bias_viz.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("[!] No bias frames found. Skipping bias calibration.")

    # Create Master Dark
    dark_files = glob.glob(os.path.join(dark_dir, "*.fit"))
    if dark_files:
        dark_stack = np.array([fits.getdata(f).astype(np.float32) for f in dark_files])
        if master_bias is not None:
            calibrated_darks = dark_stack - master_bias
            master_dark = np.median(calibrated_darks, axis=0)
        else:
            master_dark = np.median(dark_stack, axis=0)
        print(f"[*] Created Master Dark from {len(dark_files)} frames.")
        
        # Save Dark PNG
        plt.figure(figsize=(8, 8))
        vmin, vmax = np.percentile(master_dark, 1), np.percentile(master_dark, 99.9) 
        plt.imshow(master_dark, cmap='magma', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='ADU (Thermal Noise)')
        plt.title('Master Dark Frame')
        plt.savefig(os.path.join(plot_out_dir, "master_dark_viz.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        print("[!] No dark frames found. Skipping dark calibration.")
        
    return master_bias, master_dark

# --- LEVESQUE BACKGROUND SUBTRACTION ---
def levesque_process(image, sigma_psf=1.5, k=3, iterations=3):
    img_float = image.astype(np.float32)
    rows, cols = img_float.shape
    min_dim = min(rows, cols)
    
    spot_diam = 6 * sigma_psf
    raw_large = int(20 * spot_diam)
    raw_small = int(10 * spot_diam)
    
    max_window = int(min_dim / 5)
    size_large = min(raw_large, max_window)
    size_small = min(raw_small, max_window)
    
    if size_large % 2 == 0: size_large += 1
    if size_small % 2 == 0: size_small += 1
    
    background = np.full_like(img_float, np.median(img_float))
    
    for _ in range(iterations):
        residual = img_float - background
        sigma_curr = np.std(residual)
        if sigma_curr == 0: sigma_curr = 1e-6 
        
        mask = (img_float < (background + k * sigma_curr)).astype(np.float32)
        masked_img = img_float * mask
        
        smooth_numer = cv2.boxFilter(masked_img, -1, (size_large, size_large), normalize=False)
        smooth_denom = cv2.boxFilter(mask, -1, (size_large, size_large), normalize=False)
        smooth_denom[smooth_denom == 0] = 1.0
        
        bg_update = smooth_numer / smooth_denom
        background = cv2.boxFilter(bg_update, -1, (size_small, size_small), normalize=True)
        
    return img_float - background, background

# --- MAIN PIPELINE ---
def process_fits_directory(input_dir, output_csv_dir, bias_dir, dark_dir, plot_out_dir, sigma_psf=2.0, threshold_factor=10.0, skip_bg_sub=False):
    os.makedirs(output_csv_dir, exist_ok=True)
    
    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fit")))[:]
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return

    # Pre-compute Master Calibration Frames
    master_bias, master_dark = create_master_frames(bias_dir, dark_dir, plot_out_dir)

    all_detections = []
    prev_aligned = None
    
    mode_text = "SKIPPING" if skip_bg_sub else "USING"
    print(f"Processing {len(fits_files)} FITS files ({mode_text} Levesque Background Subtraction)...")

    for frame_idx, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        
        # Load FITS
        try:
            img_raw = fits.getdata(filepath).astype(np.float32)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # Apply Sensor Calibration
        img_calibrated = img_raw.copy()
        if master_bias is not None:
            img_calibrated -= master_bias
        if master_dark is not None:
            img_calibrated -= master_dark
            
        img_calibrated = np.clip(img_calibrated, a_min=0, a_max=None)

        # Background Subtraction
        if skip_bg_sub:
            img_clean = img_calibrated  
        else:
            img_clean, background_est = levesque_process(img_calibrated, sigma_psf=sigma_psf)
            
            # --- Plot and save the first Levesque execution ---
            if frame_idx == 0:
                # Calculate standard deviations for the titles
                sigma_orig = np.std(img_calibrated)
                sigma_bg = np.std(background_est)
                sigma_clean = np.std(img_clean)
                
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                # Original Calibrated Image
                vmin_orig, vmax_orig = np.percentile(img_calibrated, 1), np.percentile(img_calibrated, 99.5)
                im0 = axes[0].imshow(img_calibrated, cmap='gray', origin='lower', vmin=vmin_orig, vmax=vmax_orig)
                axes[0].set_title(f"Original (Calibrated)\n$\sigma$ = {sigma_orig:.2f} ADU")
                fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='ADU')
                
                # Background Estimate
                vmin_bg, vmax_bg = np.percentile(background_est, 1), np.percentile(background_est, 99.5)
                im1 = axes[1].imshow(background_est, cmap='viridis', origin='lower', vmin=vmin_bg, vmax=vmax_bg)
                axes[1].set_title(f"Levesque Background Est.\n$\sigma$ = {sigma_bg:.2f} ADU")
                fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='ADU')
                
                # Resulting Flattened Image
                vmin_clean, vmax_clean = np.percentile(img_clean, 1), np.percentile(img_clean, 99.5)
                im2 = axes[2].imshow(img_clean, cmap='gray', origin='lower', vmin=vmin_clean, vmax=vmax_clean)
                axes[2].set_title(f"Flattened Result\n$\sigma$ = {sigma_clean:.2f} ADU")
                fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='ADU')
                
                plt.tight_layout()
                bg_plot_path = os.path.join(plot_out_dir, "levesque_bg_sub_example.png")
                plt.savefig(bg_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[*] Saved Levesque background subtraction example to: {bg_plot_path}")

        # FIELD ROTATION CORRECTION (Astroalign)
        if frame_idx == 0:
            reference_frame = img_clean.copy()
            img_aligned = img_clean
            print("[*] Frame 1 set as alignment anchor.")
        else:
            try:
                img_aligned, footprint = aa.register(img_clean, reference_frame)
            except aa.MaxIterError:
                print(f"[!] Astroalign couldn't match stars for {filename}, skipping rotation.")
                img_aligned = img_clean 
                
        # --- IMAGE DIFFERENCING ---
        if prev_aligned is None:
            prev_aligned = img_aligned.copy()
            print(f"Processed frame {frame_idx + 1} / {len(fits_files)} (Stored as diff reference)")
            continue
            
        diff_img = img_aligned - prev_aligned
        prev_aligned = img_aligned.copy()
        diff_positive = np.clip(diff_img, a_min=0, a_max=None)
        score_map = cv2.GaussianBlur(diff_positive, (5, 5), sigmaX=1.5)
        
        # Robust Thresholding
        median_score = np.median(score_map)
        robust_sigma = 1.4826 * np.median(np.abs(score_map - median_score))
        if robust_sigma == 0: robust_sigma = 1e-6 
        
        threshold = threshold_factor * robust_sigma
        bright_indices = np.argwhere(score_map > threshold)
        
        # Clustering & Centroiding
        if len(bright_indices) > 0:
            clustering = DBSCAN(eps=5.0, min_samples=3).fit(bright_indices)
            labels = clustering.labels_
            unique_labels = set(labels) - {-1} 
            
            for label in unique_labels:
                cluster_mask = (labels == label)
                points = bright_indices[cluster_mask]
                ys, xs = points[:, 0], points[:, 1]
                
                cluster_scores = score_map[ys, xs]
                total_score = np.sum(cluster_scores)
                
                if total_score > 0:
                    y_c = np.sum(ys * cluster_scores) / total_score
                    x_c = np.sum(xs * cluster_scores) / total_score
                    
                    peak_val = np.max(cluster_scores)
                    snr = peak_val / robust_sigma
                    
                    pos_variance = (sigma_psf / snr)**2 if snr > 0 else 0.0
                    
                    det_record = {
                        'Frame_Idx': frame_idx,
                        'Filename': filename,
                        'Centroid_X': x_c,
                        'Centroid_Y': y_c,
                        'Cov_XX': pos_variance,
                        'Cov_YY': pos_variance,
                        'Cov_XY': 0.0, 
                        'SNR': snr,
                        'Peak_Value': peak_val,
                        'Cluster_Size': len(points)
                    }
                    all_detections.append(det_record)
                    
        if (frame_idx + 1) % 10 == 0:
            print(f"Processed frame {frame_idx + 1} / {len(fits_files)}")

    # --- EXPORT TO CSV ---
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_path = os.path.join(output_csv_dir, "master_detections_with_covariance.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDetection complete! {len(df)} total target detections saved to: {csv_path}")
    else:
        print("\nNo detections found above the threshold in any frame.")

if __name__ == "__main__":
    INPUT_FITS_DIR = r"D:\tanne\Documents\mht_dataset\multi_target"
    OUTPUT_CSV_DIR = "results/pipeline_output"
    PLOT_OUT_DIR = "results/calibration_plots"
    BIAS_DIR = r"D:\tanne\Documents\mht_dataset\bias"
    DARK_DIR = r"D:\tanne\Documents\mht_dataset\dark"
    
    # Run the pipeline
    process_fits_directory(
        input_dir=INPUT_FITS_DIR, 
        output_csv_dir=OUTPUT_CSV_DIR, 
        bias_dir=BIAS_DIR,
        dark_dir=DARK_DIR,
        plot_out_dir=PLOT_OUT_DIR,
        sigma_psf=4.0, 
        threshold_factor=100.0,
        skip_bg_sub=False
    )
    