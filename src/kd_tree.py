import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def draw_kdtree_partitions(ax, points, depth, x_min, x_max, y_min, y_max, max_depth=5):
    """
    Recursively draws the orthogonal splitting planes of a 2D KD-Tree.
    """
    # Stop condition: no points left or max depth reached
    if len(points) == 0 or depth >= max_depth:
        return
    
    # Alternating axes: 0 for X-axis (vertical cut), 1 for Y-axis (horizontal cut)
    axis = depth % 2 
    
    # Sort points along the current active axis to find the median
    sorted_points = points[points[:, axis].argsort()]
    median_idx = len(sorted_points) // 2
    median_val = sorted_points[median_idx, axis]
    
    if axis == 0:
        # Split on X: draw a vertical line
        ax.plot([median_val, median_val], [y_min, y_max], color='red', linestyle='-', linewidth=1.2, alpha=0.8)
        
        # Recurse Left
        draw_kdtree_partitions(ax, sorted_points[:median_idx], depth + 1, x_min, median_val, y_min, y_max, max_depth)
        # Recurse Right
        draw_kdtree_partitions(ax, sorted_points[median_idx+1:], depth + 1, median_val, x_max, y_min, y_max, max_depth)
    else:
        # Split on Y: draw a horizontal line
        ax.plot([x_min, x_max], [median_val, median_val], color='blue', linestyle='-', linewidth=1.2, alpha=0.8)
        
        # Recurse Bottom
        draw_kdtree_partitions(ax, sorted_points[:median_idx], depth + 1, x_min, x_max, y_min, median_val, max_depth)
        # Recurse Top
        draw_kdtree_partitions(ax, sorted_points[median_idx+1:], depth + 1, x_min, x_max, median_val, y_max, max_depth)


def visualize_frame_kdtree(csv_path: str, output_path: str, max_tree_depth: int = 5):
    # 1. Load data and extract the first frame
    df = pd.read_csv(csv_path)
    # Rename columns just in case, matching your previous pipeline logic
    if 'Frame_Idx' in df.columns:
        df = df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})
        
    first_frame_time = df['time'].min()
    frame_data = df[df['time'] == first_frame_time][['x', 'y']].to_numpy()
    
    if len(frame_data) == 0:
        print("No data found for the first frame!")
        return

    # 2. Setup the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get spatial boundaries with a small buffer
    buffer = 20
    x_min, x_max = frame_data[:, 0].min() - buffer, frame_data[:, 0].max() + buffer
    y_min, y_max = frame_data[:, 1].min() - buffer, frame_data[:, 1].max() + buffer
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # 3. Draw the KD-Tree partitions recursively
    draw_kdtree_partitions(ax, frame_data, depth=0, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, max_depth=max_tree_depth)
    
    # 4. Scatter the actual detection points on top
    ax.scatter(frame_data[:, 0], frame_data[:, 1], c='black', s=30, zorder=5, label='Frame 1 Detections')
    
    # 5. Formatting
    ax.set_title(f"KD-Tree Spatial Partitioning (Frame {first_frame_time})", fontsize=14)
    ax.set_xlabel("X Position (Pixels)")
    ax.set_ylabel("Y Position (Pixels)")
    
    # Custom legend for the split lines
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='blue', lw=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8)
    ]
    ax.legend(custom_lines, ['X-Axis Splits', 'Y-Axis Splits', 'Detections'], loc='upper right')
    
    ax.grid(False) # Turn off standard grid so KD-tree lines pop
    plt.tight_layout()
    
    # 6. Save and show
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"KD-Tree visualization saved to {output_path}")

if __name__ == "__main__":
    # Update these paths to match your directory structure
    input_csv = "results/pipeline_output/master_detections_with_covariance.csv"
    output_image = "results/tomht_eval/kdtree_visualization.png"
    
    # You can tweak max_tree_depth. 
    # Depth 4-6 usually looks best for explaining the concept without creating a chaotic grid.
    visualize_frame_kdtree(input_csv, output_image, max_tree_depth=25)