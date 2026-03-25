import pandas as pd
import matplotlib.pyplot as plt

def plot_longest_tracks(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str):
    """Plots all raw detections and overlays the top 5 longest tracker continuous paths."""
    plt.figure(figsize=(12, 8))
    
    # 1. Scatter all raw detections in the background
    plt.scatter(meas_df['x'], meas_df['y'], c='lightgray', s=10, alpha=0.5, label='Raw Detections')
    
    # 2. Draw the tracks
    if not tracked_df.empty:
        # Find the 5 "best" tracks (the ones with the most frames/longest history)
        track_lengths = tracked_df.groupby('id').size()
        top_5_track_ids = track_lengths.nlargest(5).index
        
        # Group by track ID and plot lines only for the top 5
        for track_id, grp in tracked_df.groupby('id'):
            if track_id in top_5_track_ids: 
                plt.plot(grp['x'], grp['y'], marker='.', markersize=4, linewidth=1.5, label=f'Track {track_id}')
    
    plt.title("TOMHT Output vs. Raw Detections (Top 5 Tracks)")
    plt.xlabel("X Position (Pixels)")
    plt.ylabel("Y Position (Pixels)")
    
    # Show legend (no need to slice handles anymore since there are max 5 + 1 items)
    if not tracked_df.empty:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved static plot to {output_path}")