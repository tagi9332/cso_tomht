import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any

def plot_frame_grid(
    frames: List[Any], 
    n: int, 
    output_dir: str, 
    vmin: float, 
    vmax: float,
    title: str = "Simulation Frame Progression",
    step: int = 2
) -> None:
    """
    Generates and saves an n x n grid plot of the frames.
    Frames progress from left to right, top to bottom.
    
    Args:
        frames: List of image arrays.
        n: Dimension of the grid (n x n).
        output_dir: Directory to save the plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        title: Descriptive title for the overall figure.
        step: Interval for selecting frames (e.g., 2 plots every 2nd frame).
    """
    # Update title to reflect the step size if it's greater than 1
    if step == 2:
        title = f"{title} (Every {step}nd Frame)"
    elif step == 3:
        title = f"{title} (Every {step}rd Frame)"
    else:
        title = f"{title} (Every {step}th Frame)"

    num_frames = n * n
    # Extract the required number of frames applying the step size
    frames_subset = frames[0 : num_frames * step : step]
    
    # Scale figure size based on grid dimensions
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    
    # Add the overall descriptive title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Handle the n=1 edge case and flatten for linear iteration
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(frames_subset):
            ax.imshow(
                frames_subset[idx], 
                cmap='gray', 
                origin='lower', 
                interpolation='nearest', 
                vmin=vmin, 
                vmax=vmax
            )
            # Calculate and display the original frame index
            original_frame_idx = idx * step
            ax.set_title(f"Frame {original_frame_idx}", fontsize=10)
        else:
            # Hide axes if the list has fewer frames than the grid requires
            ax.axis('off')
            
        # Remove axis ticks for a cleaner appearance in reports
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Adjust layout to prevent the title from overlapping with the subplots
    plt.tight_layout()
    fig.subplots_adjust(top=0.92) 
    
    plt.savefig(os.path.join(output_dir, f"simulation_frame_grid_{n}x{n}.png"), dpi=200)
    plt.close(fig)