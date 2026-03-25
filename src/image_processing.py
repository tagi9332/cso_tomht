import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN
from PIL import Image

# --- 1. LEVESQUE BACKGROUND SUBTRACTION ---
def levesque_process(image, sigma_psf=2.0, k=2.5, iterations=5):
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

# --- 2. MAIN PIPELINE ---
def process_fits_directory(input_dir, output_csv_dir, output_plot_dir, sigma_psf=2.0, threshold_factor=5.0, skip_bg_sub=False):
    # Ensure output directories exist
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return

    # Pre-compute Matched Filter Kernel
    kernel_rad = int(3 * sigma_psf)
    y_k, x_k = np.mgrid[-kernel_rad:kernel_rad+1, -kernel_rad:kernel_rad+1]
    psf_kernel = np.exp(-(x_k**2 + y_k**2) / (2 * sigma_psf**2))
    psf_kernel /= np.sum(psf_kernel)
    
    all_detections = []
    saved_plot_paths = []
    
    mode_text = "SKIPPING" if skip_bg_sub else "USING"
    print(f"Processing {len(fits_files)} FITS files ({mode_text} Levesque Background Subtraction)...")

    for frame_idx, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        # 1. Load FITS
        try:
            img_raw = fits.getdata(filepath).astype(np.float32)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # 2. Background Subtraction Logic
        if skip_bg_sub:
            img_clean = img_raw  # Bypass subtraction
        else:
            img_clean, background_est = levesque_process(img_raw, sigma_psf=sigma_psf)

            # --- NEW: SAVE DIAGNOSTIC COMPARISON (For the first 3 frames) ---
        if frame_idx % 20 == 0:
            diag_dir = os.path.join(output_plot_dir, "background_diagnostics")
            os.makedirs(diag_dir, exist_ok=True)
            
            fig_diag, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Raw Image
            axes[0].imshow(img_raw, cmap='gray', origin='lower')
            axes[0].set_title("1. Raw Input")
            
            # Estimated Background
            axes[1].imshow(background_est, cmap='magma', origin='lower')
            axes[1].set_title("2. Levesque Est. Background")
            
            # Cleaned Image (Residual)
            axes[2].imshow(img_clean, cmap='gray', origin='lower')
            axes[2].set_title("3. Cleaned (Residual)")
            
            for ax in axes: ax.axis('off')
            
            diag_save_path = os.path.join(diag_dir, f"{base_name}_bg_diag.png")
            plt.savefig(diag_save_path, dpi=150)
            plt.close(fig_diag)
            print(f"Saved background diagnostic to {diag_save_path}")
        
        # 3. Matched Filter
        score_map = convolve2d(img_clean, psf_kernel, mode='same')
        
        # 4. Robust Thresholding
        median_score = np.median(score_map)
        robust_sigma = 1.4826 * np.median(np.abs(score_map - median_score))
        if robust_sigma == 0: robust_sigma = 1e-6 
        
        threshold = threshold_factor * robust_sigma
        bright_indices = np.argwhere(score_map > threshold)
        
        frame_detections = []
        
        # 5. Clustering & Centroiding
        if len(bright_indices) > 0:
            clustering = DBSCAN(eps=3.0, min_samples=2).fit(bright_indices)
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
                    frame_detections.append(det_record)
                    all_detections.append(det_record)
                    
        # --- 6. Plotting and Saving PNG ---
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Stretch contrast so background isn't pure black and bright spots don't blow out
        vmin, vmax = np.min(img_raw), np.max(img_raw)
        ax.imshow(img_raw, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        
        # Overlay detections
        for det in frame_detections:
            x, y = det['Centroid_X'], det['Centroid_Y']
            ax.plot(x, y, 'r+', markersize=10, markeredgewidth=1.5)
            circle = plt.Circle((x, y), 3 * sigma_psf, color='red', fill=False, linestyle='--', alpha=0.7)
            ax.add_patch(circle)
            ax.text(x + 5, y + 5, f"SNR:{det['SNR']:.1f}", color='yellow', fontsize=8)

        ax.set_title(f"Frame: {filename} | Detections: {len(frame_detections)}")
        ax.axis('off') 
        
        plot_save_path = os.path.join(output_plot_dir, f"{base_name}_detections.png")
        plt.tight_layout()
        plt.savefig(plot_save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
        saved_plot_paths.append(plot_save_path)

        if (frame_idx + 1) % 10 == 0:
            print(f"Processed frame {frame_idx + 1} / {len(fits_files)}")

    # --- 7. EXPORT TO CSV & GIF ---
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_path = os.path.join(output_csv_dir, "master_detections_with_covariance.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDetection complete! {len(df)} total detections saved to: {csv_path}")
    else:
        print("\nNo detections found above the threshold in any frame.")
        
    if saved_plot_paths:
        gif_path = os.path.join(output_plot_dir, "detections_animation.gif")
        print(f"Generating GIF animation at {gif_path}...")
        
        frames = [Image.open(p) for p in saved_plot_paths]
        frames[0].save(
            gif_path,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=200, 
            loop=0 
        )
        print("GIF generation complete!")

if __name__ == "__main__":
    INPUT_FITS_DIR = "results/fits" 
    OUTPUT_CSV_DIR = "results/pipeline_output"
    OUTPUT_PLOT_DIR = "results/detection_plots" 
    
    # Run the pipeline
    process_fits_directory(
        input_dir=INPUT_FITS_DIR, 
        output_csv_dir=OUTPUT_CSV_DIR, 
        output_plot_dir=OUTPUT_PLOT_DIR,
        sigma_psf=2.0, 
        threshold_factor=4.0,
        skip_bg_sub=False
    )