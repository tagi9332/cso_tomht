import os
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import List, Any

def export_frames(
    frames: List[Any], 
    output_dir: str, 
    vmin: float, 
    vmax: float
) -> None:
    """Exports raw simulation frames to FITS and scaled PNG formats."""
    print("Exporting individual frames to FITS and PNG...")
    
    fits_dir = os.path.join(output_dir, "fits")
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(fits_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    for i, frame in enumerate(frames):
        # Save as FITS (raw float/int data)
        hdu = fits.PrimaryHDU(frame)
        fits_filename = os.path.join(fits_dir, f"frame_{i:04d}.fits")
        hdu.writeto(fits_filename, overwrite=True)
        
        # Save as PNG (scaled grayscale image for easy viewing)
        png_filename = os.path.join(png_dir, f"frame_{i:04d}.png")
        plt.imsave(png_filename, frame, cmap='gray', vmin=vmin, vmax=vmax)

    print(f"Exported {len(frames)} frames.")