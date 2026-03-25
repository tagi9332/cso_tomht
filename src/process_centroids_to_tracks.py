import pandas as pd
import os
from typing import cast
from utils.config_loader import TrackerConfig
from src.tomht import TOMHTTracker

def process_centroids_to_tracks(csv_path: str) -> pd.DataFrame:
    """Reads centroid data, runs TOMHT tracking, and returns delayed states."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Centroid CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={'Frame_Idx': 'time', 'Centroid_X': 'x', 'Centroid_Y': 'y'})
    
    config = TrackerConfig.from_jsonx("config/parameters.jsonx")
    tracker = TOMHTTracker(config)
    n_scan = config.n_scan_window 
    results = []
    
    for t_raw, group in df.groupby('time'):
        t = cast(int, t_raw)
        meas_t = group[['x', 'y']].to_numpy()
        active_tracks = tracker.step(meas_t)

        for track in active_tracks:
            if len(track.best_hypothesis.history_states) >= n_scan: 
                delayed_state = track.best_hypothesis.history_states[-n_scan]
                results.append({
                    'time': t - n_scan,
                    'id': track.track_id,
                    'x': delayed_state[0], 'y': delayed_state[1],
                    'vx': delayed_state[2], 'vy': delayed_state[3]
                })
                
    return pd.DataFrame(results)