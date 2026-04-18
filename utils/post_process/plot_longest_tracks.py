import pandas as pd
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