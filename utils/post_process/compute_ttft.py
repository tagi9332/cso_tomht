import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

def compute_ttft(gt_df: pd.DataFrame, trk_df: pd.DataFrame, distance_threshold: float = 5.0, verbose: bool = True) -> pd.DataFrame:
    """
    Calculates TTFT, Unique Track IDs, and Fragmentation (ID Switches) 
    using continuous frame-by-frame Hungarian matching.
    """
    # Find the birth frame for every ground truth object
    birth_times = gt_df.groupby('Object_ID')['Frame'].min().to_dict()
    
    # Dictionary to store the matching history
    match_history = {obj_id: [] for obj_id in birth_times.keys()}
    
    all_frames = sorted(gt_df['Frame'].unique())
    
    # Perform 1-to-1 Hungarian matching on every single frame
    for f in all_frames:
        gt_f = gt_df[gt_df['Frame'] == f]
        trk_f = trk_df[trk_df['time'] == f]
        
        if gt_f.empty or trk_f.empty:
            continue
            
        # Calculate distances between all GT and Tracks in this specific frame
        dist_matrix = cdist(gt_f[['True_X', 'True_Y']], trk_f[['x', 'y']])
        gt_indices, trk_indices = linear_sum_assignment(dist_matrix)
        
        for r, c in zip(gt_indices, trk_indices):
            if dist_matrix[r, c] <= distance_threshold:
                obj_id = gt_f.iloc[r]['Object_ID']
                trk_id = trk_f.iloc[c]['id']
                # Record a successful match for this object at this frame
                match_history[obj_id].append((f, trk_id))
                
    # Process the histories to extract TTFT and Fragmentation metrics
    results = []
    for obj_id, birth_frame in birth_times.items():
        matches = match_history[obj_id]
        
        if not matches:
            results.append({
                'Object_ID': obj_id,
                'Birth Frame': int(birth_frame),
                'First Tracked Frame': 'N/A',
                'TTFT (Frames)': 'N/A',
                'Initial Track ID': 'N/A',
                'Unique Track IDs': 0,
                'Frags / ID Switches': 0,
                'Status': 'Missed completely'
            })
        else:
            # Ensure chronological order
            matches.sort(key=lambda x: x[0])
            
            first_tracked_frame = matches[0][0]
            initial_trk_id = matches[0][1]
            ttft = first_tracked_frame - birth_frame
            
            # Calculate Unique Track IDs
            assigned_track_ids = [m[1] for m in matches]
            unique_track_ids = len(set(assigned_track_ids))
            
            # Calculate ID Switches / Fragmentation Events
            id_switches = 0
            prev_trk_id = assigned_track_ids[0]
            for current_trk_id in assigned_track_ids[1:]:
                if current_trk_id != prev_trk_id:
                    id_switches += 1
                    prev_trk_id = current_trk_id
                    
            results.append({
                'Object_ID': obj_id,
                'Birth Frame': int(birth_frame),
                'First Tracked Frame': int(first_tracked_frame),
                'TTFT (Frames)': int(ttft),
                'Initial Track ID': int(initial_trk_id),
                'Unique Track IDs': int(unique_track_ids),
                'Frags / ID Switches': int(id_switches),
                'Status': 'Tracked'
            })
            
    results_df = pd.DataFrame(results)
    
    # Print console table
    if verbose and not results_df.empty:
        print("\n" + "="*95)
        print(" TRACK LIFESPAN, INITIALIZATION, AND FRAGMENTATION SUMMARY")
        print("="*95)
        print(tabulate(results_df, headers='keys', tablefmt='psql', showindex=False))
        
        tracked_only = results_df[results_df['Status'] == 'Tracked']
        if not tracked_only.empty:
            avg_ttft = tracked_only['TTFT (Frames)'].mean()
            avg_frags = tracked_only['Frags / ID Switches'].mean()
            print(f"\n[*] Average Time Until Tracked : {avg_ttft:.2f} frames")
            print(f"[*] Average Frags/Switches per Target: {avg_frags:.2f} breaks")
        print("="*95 + "\n")
            
    return results_df