import pandas as pd

def print_tomht_stats(meas_df: pd.DataFrame, tracked_df: pd.DataFrame):
    """Prints a summary of the TOMHT tracker's performance."""
    print("\n" + "="*50)
    print("             TOMHT PERFORMANCE SUMMARY")
    print("="*50)
    
    # Basic Counts
    total_frames = meas_df['time'].nunique()
    total_detections = len(meas_df)
    total_track_points = len(tracked_df)
    unique_tracks = tracked_df['id'].nunique() if not tracked_df.empty else 0
    
    print(f"Frames Processed      : {total_frames}")
    print(f"Raw Detections        : {total_detections}")
    print(f"Total Tracked Points  : {total_track_points}")
    print(f"Unique Tracks Formed  : {unique_tracks}")
    
    print("-" * 50)
    print(" TRACK LIFESPAN METRICS")
    print("-" * 50)
    
    if unique_tracks > 0:
        track_lengths = tracked_df.groupby('id').size()
        print(f"Max Track Length      : {track_lengths.max()} frames")
        print(f"Average Track Length  : {track_lengths.mean():.1f} frames")
        
        # Categorize tracks by length
        short_tracks = (track_lengths <= 5).sum()
        med_tracks = ((track_lengths > 5) & (track_lengths <= 20)).sum()
        long_tracks = (track_lengths > 20).sum()
        
        print(f"Short Tracks (<=5)    : {short_tracks}")
        print(f"Medium Tracks (6-20)  : {med_tracks}")
        print(f"Long Tracks (>20)     : {long_tracks}")
    else:
        print("No tracks were formed!")
        
    print("="*50 + "\n")