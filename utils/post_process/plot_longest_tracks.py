import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_longest_tracks(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str, display_plots: bool = False, n_tracks: int = 10):
    """Plots all raw detections and overlays the top 5 longest tracker continuous paths."""
    plt.figure(figsize=(12, 8))
    
    # Scatter all raw detections in the background
    plt.scatter(meas_df['x'], meas_df['y'], c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # Draw the tracks
    if not tracked_df.empty:
        # Find the 5 "best" tracks (the ones with the most frames/longest history)
        track_lengths = tracked_df.groupby('id').size()
        top_n_track_ids = track_lengths.nlargest(n_tracks).index
        
        # Group by track ID and plot lines only for the top 5
        for track_id, grp in tracked_df.groupby('id'):
            if track_id in top_n_track_ids: 
                plt.plot(grp['x'], grp['y'], marker='.', markersize=4, linewidth=1.5, label=f'Track {track_id}')
    
    plt.title(f"TOMHT Output vs. Raw Detections (Top {n_tracks} Longest Tracks)")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    # Show legend
    if not tracked_df.empty:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if display_plots:
        plt.show()
    plt.close()

    
    print(f"Saved track plot to {output_path}")


def plot_longest_tracks_cwh(
    meas_df: pd.DataFrame, 
    tracked_df: pd.DataFrame, 
    output_path: str, 
    display_plots: bool = False, 
    n_tracks: int = 20,
    cx: float = 256,           
    cy: float = 256,           
    pixel_pitch: float = 1.5e-6,  
    focal_length: float = 4.0,
    camera_offset_x: float = -1000000.0  # <-- ADD THIS: Must match your EKF config!
):
    """Plots all raw detections and overlays the top n longest tracker paths projected back to pixels."""
    plt.figure(figsize=(12, 8))
    
    # 1. Scatter all raw detections in the background
    if 'Centroid_X' in meas_df.columns and 'Centroid_Y' in meas_df.columns:
        raw_px = meas_df['Centroid_X']
        raw_py = meas_df['Centroid_Y']
    else:
        raw_px = (meas_df['u'] / pixel_pitch) + cx
        raw_py = (meas_df['v'] / pixel_pitch) + cy
        
    plt.scatter(raw_px, raw_py, c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # 2. Draw the tracks mapped back to the pixel plane
    if not tracked_df.empty:
        track_lengths = tracked_df.groupby('id').size()
        top_n_track_ids = track_lengths.nlargest(n_tracks).index
        
        for track_id, grp in tracked_df.groupby('id'):
            if track_id in top_n_track_ids: 
                # --- CRITICAL FIX: Calculate range relative to the camera ---
                range_x = grp['x'] - camera_offset_x
                
                # Prevent divide-by-zero just in case
                range_x = np.where(range_x == 0, 1e-6, range_x)
                
                u_pred_meters = focal_length * (grp['y'] / range_x)
                v_pred_meters = focal_length * (grp['z'] / range_x)
                
                # --- CRITICAL FIX: Cleaned up the X/Y sign flipping logic ---
                # Adjust these + / - signs based on exactly how your sensor frame maps to CWH
                track_px = cx - (u_pred_meters / pixel_pitch) 
                track_py = cy - (v_pred_meters / pixel_pitch) 
                
                plt.plot(track_px, track_py, marker='.', markersize=4, linewidth=1.5, label=f'Track {track_id}')
    
    plt.title(f"TOMHT Output vs. Raw Detections (Top {n_tracks} Longest Tracks)")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    if not tracked_df.empty:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.axis('equal')
    
    if display_plots:
        plt.show()
        
    plt.close()
    print(f"[*] Saved track plot to {output_path}")