import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_motp(gt_df: pd.DataFrame, trk_df: pd.DataFrame, distance_threshold: float = 5.0) -> float:
    """
    Calculates the Multiple Object Tracking Precision (MOTP).
    Measures the average localization error for successfully matched targets.
    """
    total_distance = 0.0
    total_matches = 0
    
    # Iterate through every frame present in both datasets
    common_frames = set(gt_df['Frame']).intersection(set(trk_df['time']))
    
    for f in common_frames:
        gt_f = gt_df[gt_df['Frame'] == f]
        trk_f = trk_df[trk_df['time'] == f]
        
        # Calculate distance matrix between all GT and Track coordinates
        dist_matrix = cdist(gt_f[['True_X', 'True_Y']], trk_f[['x', 'y']])
        
        # Hungarian algorithm for optimal 1-to-1 matching
        gt_indices, trk_indices = linear_sum_assignment(dist_matrix)
        
        for r, c in zip(gt_indices, trk_indices):
            dist = dist_matrix[r, c]
            if dist <= distance_threshold:
                total_distance += dist
                total_matches += 1
                
    if total_matches == 0:
        return 0.0 # Avoid division by zero
        
    motp = float(total_distance / total_matches)
    return motp

def compute_mota(gt_df: pd.DataFrame, trk_df: pd.DataFrame, distance_threshold: float = 5.0) -> float:
    """
    Calculates the Multiple Object Tracking Accuracy (MOTA).
    Combines False Positives (FP), False Negatives (FN), and ID Switches (IDSW).
    """
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    total_gt = len(gt_df)
    
    # Dictionary to keep track of GT_ID -> Track_ID mappings across frames to detect ID switches
    gt_to_trk_mapping = {}
    
    all_frames = sorted(set(gt_df['Frame']).union(set(trk_df['time'])))
    
    for f in all_frames:
        gt_f = gt_df[gt_df['Frame'] == f]
        trk_f = trk_df[trk_df['time'] == f]
        
        if gt_f.empty:
            total_fp += len(trk_f) # All tracks here are false positives
            continue
            
        if trk_f.empty:
            total_fn += len(gt_f) # All GT here are false negatives/misses
            continue
            
        dist_matrix = cdist(gt_f[['True_X', 'True_Y']], trk_f[['x', 'y']])
        gt_indices, trk_indices = linear_sum_assignment(dist_matrix)
        
        matched_gt_indices = set()
        matched_trk_indices = set()
        
        for r, c in zip(gt_indices, trk_indices):
            if dist_matrix[r, c] <= distance_threshold:
                matched_gt_indices.add(r)
                matched_trk_indices.add(c)
                
                gt_id = gt_f.iloc[r]['Object_ID']
                trk_id = trk_f.iloc[c]['id']
                
                # Check for Identity Switch (IDSW)
                if gt_id in gt_to_trk_mapping:
                    if gt_to_trk_mapping[gt_id] != trk_id:
                        total_idsw += 1
                
                # Update mapping
                gt_to_trk_mapping[gt_id] = trk_id
                
        # Update FP and FN
        total_fn += len(gt_f) - len(matched_gt_indices)
        total_fp += len(trk_f) - len(matched_trk_indices)
        
    if total_gt == 0:
        return 0.0
        
    mota = 1.0 - ((total_fn + total_fp + total_idsw) / total_gt)
    return mota