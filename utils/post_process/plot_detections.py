import os
import numpy as np
import matplotlib.pyplot as plt

def plot_background_diagnostics(img_raw, background_est, img_clean, output_dir, base_name, display_plots=False):
    """Saves a 3-panel plot comparing raw, background, and cleaned images."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_raw, cmap='gray', origin='lower')
    axes[0].set_title("1. Raw Input")
    
    axes[1].imshow(background_est, cmap='magma', origin='lower')
    axes[1].set_title("2. Est. Background")
    
    axes[2].imshow(img_clean, cmap='gray', origin='lower')
    axes[2].set_title("3. Cleaned (Residual)")
    
    for ax in axes: ax.axis('off')
    
    save_path = os.path.join(output_dir, f"{base_name}_bg_diag.png")
    plt.savefig(save_path, dpi=150)
    if display_plots:
        plt.show()
    plt.close(fig)

def plot_detections(img_raw, detections, sigma_psf, output_dir, base_name, filename, display_plots=False):
    """Overlays detection circles and SNR text on the raw image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    vmin, vmax = np.min(img_raw), np.max(img_raw)
    ax.imshow(img_raw, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    
    for det in detections:
        x, y = det['Centroid_X'], det['Centroid_Y']
        ax.plot(x, y, 'r+', markersize=10, markeredgewidth=1.5)
        circle = plt.Circle((x, y), 3 * sigma_psf, color='red', fill=False, linestyle='--', alpha=0.7) # type: ignore
        ax.add_patch(circle)
        ax.text(x + 5, y + 5, f"SNR:{det['SNR']:.1f}", color='yellow', fontsize=8)

    ax.set_title(f"Frame: {filename} | Detections: {len(detections)}")
    ax.axis('off') 
    
    save_path = os.path.join(output_dir, f"{base_name}_detections.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if display_plots:
        plt.show()
    plt.close(fig) 
    
    return save_path