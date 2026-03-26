import os
import glob
import pandas as pd
import numpy as np
from astropy.io import fits
from PIL import Image
import json
import re

# Local Utility Imports
from utils.background_subtraction.levesque_bkgnd_subtractor import levesque_process
from utils.detector_filters.matched_filter import create_psf_kernel, detect_sources
from utils.post_process.plot_detections import plot_detections

def process_fits_directory(
    input_dir, 
    output_csv_dir, 
    output_plot_dir, 
    sigma_psf=2.0, 
    threshold_factor=4.0, 
    skip_bg_sub=False,
    verbose=False,
    generate_gif=False
):
    """
    Cleans frames and extracts source detections.
    """
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    fits_files = sorted(glob.glob(os.path.join(input_dir, "*.fits")))
    if not fits_files:
        if verbose: print(f"[!] No FITS files found in {input_dir}")
        return

    psf_kernel = create_psf_kernel(sigma_psf)
    all_detections = []
    saved_plot_paths = []
    
    if verbose:
        mode = "SKIPPING" if skip_bg_sub else "USING"
        print(f"[*] Starting Detection: {len(fits_files)} frames ({mode} BKG Subtraction)")

    for frame_idx, filepath in enumerate(fits_files):
        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        
        try:
            img_raw = fits.getdata(filepath).astype(np.float32)
        except Exception as e:
            if verbose: print(f"\n[!] Error reading {filename}: {e}")
            continue
            
        # Processing logic
        if skip_bg_sub:
            img_clean = img_raw
        else:
            img_clean, _ = levesque_process(img_raw, sigma_psf=sigma_psf)

        frame_detections = detect_sources(
            img_clean=img_clean,
            psf_kernel=psf_kernel,
            sigma_psf=sigma_psf,
            threshold_factor=threshold_factor,
            frame_idx=frame_idx,
            filename=filename
        )
        all_detections.extend(frame_detections)
        
        # Save visualization paths for GIF
        plot_save_path = plot_detections(img_raw, frame_detections, sigma_psf, output_plot_dir, base_name, filename)
        saved_plot_paths.append(plot_save_path)

        # Interactive Progress Update
        if verbose:
            print(f"\r[+] Processing Frame {frame_idx + 1}/{len(fits_files)}", end='', flush=True)


    # Finalization
    if not verbose: print()
    
    if all_detections:
        df = pd.DataFrame(all_detections)
        csv_path = os.path.join(output_csv_dir, "master_detections_with_covariance.csv")
        df.to_csv(csv_path, index=False)
        if verbose: print(f"[✓] Saved {len(df)} detections to CSV.")
    
    if saved_plot_paths:
        if generate_gif:
            gif_path = os.path.join(output_plot_dir, "detections_animation.gif")
            if verbose: print(f"[*] Building animation...")
            
            frames = [Image.open(p) for p in saved_plot_paths]
            frames[0].save(
                gif_path, format='GIF', append_images=frames[1:],
                save_all=True, duration=200, loop=0 
            )
            if verbose: print(f"[✓] GIF saved to {gif_path}")

if __name__ == "__main__":
    process_fits_directory(
        input_dir="results/fits", 
        output_csv_dir="results/pipeline_output",
        output_plot_dir="results/detection_plots",
        verbose=True 
    )