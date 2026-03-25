import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
from PIL import Image

# Local Utility Imports
from utils.background_subtraction.levesque_bkgnd_subtractor import levesque_process
from utils.detector_filters.matched_filter import create_psf_kernel, detect_sources
from utils.post_process.plot_detections import plot_background_diagnostics, plot_detections

#  Main Processing Function
def process_fits_directory(input_dir, output_csv_dir, output_plot_dir, sigma_psf=2.0, threshold_factor=5.0, skip_bg_sub=False):
    # Ensure output directories exist
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if not fits_files:
        print(f"No FITS files found in {input_dir}")
        return

    # Pre-compute Matched Filter Kernel
    psf_kernel = create_psf_kernel(sigma_psf)
    
    all_detections = []
    saved_plot_paths = []
    
    mode_text = "SKIPPING" if skip_bg_sub else "USING"
    print(f"Processing {len(fits_files)} FITS files ({mode_text} Levesque Background Subtraction)...")

    for frame_idx, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        # Load FITS
        try:
            img_raw = fits.getdata(filepath).astype(np.float32)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
            
        # Background Subtraction (if not skipped)
        if skip_bg_sub:
            img_clean = img_raw
        else:
            img_clean, background_est = levesque_process(img_raw, sigma_psf=sigma_psf)

            # # DEBUG: Save Background Diagnostic Plot (For the first and every 20th frame)
            # if frame_idx % 20 == 0:
            #     diag_dir = os.path.join(output_plot_dir, "background_diagnostics")
            #     plot_background_diagnostics(img_raw, background_est, img_clean, diag_dir, base_name)
        
        # Matched Filter & Detection
        frame_detections = detect_sources(
            img_clean=img_clean,
            psf_kernel=psf_kernel,
            sigma_psf=sigma_psf,
            threshold_factor=threshold_factor,
            frame_idx=frame_idx,
            filename=filename
        )
        all_detections.extend(frame_detections)
                    
        # Plotting and Saving PNG
        plot_save_path = plot_detections(img_raw, frame_detections, sigma_psf, output_plot_dir, base_name, filename)
        saved_plot_paths.append(plot_save_path)

        if (frame_idx + 1) % 10 == 0:
            print(f"Processed frame {frame_idx + 1} / {len(fits_files)}")

    # Export to CSV
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_path = os.path.join(output_csv_dir, "master_detections_with_covariance.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nDetection complete! {len(df)} total detections saved to: {csv_path}")
    else:
        print("\nNo detections found above the threshold in any frame.")
        
    # Generate GIF Animation
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