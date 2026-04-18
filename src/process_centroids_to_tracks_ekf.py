import pandas as pd
import os
import sys
from typing import cast
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from utils.config_loader import TrackerConfig
from src.tomht_ekf import TOMHTTracker

def process_centroids_to_tracks(csv_path = None):
    '''Reads centroid data, runs TOMHT tracking, and returns 6D delayed states.'''
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'''Centroid CSV not found at {csv_path}''')
    df = pd.read_csv(csv_path)
    df = df.rename(columns = {
        'Frame_Idx': 'time',
        'Centroid_X': 'u',
        'Centroid_Y': 'v' })
    config = TrackerConfig.from_jsonx('config/parameters.jsonx')
    tracker = TOMHTTracker(config)
    n_scan = config.n_scan_window
    results = []
    for t_raw, group in df.groupby('time'):
        t = cast(int, t_raw)
        meas_t = group[[
            'u',
            'v']].to_numpy()
        active_tracks = tracker.step(meas_t)
        for track in active_tracks:
            if not len(track.best_hypothesis.history_states) >= n_scan:
                continue
            delayed_state = track.best_hypothesis.history_states[-n_scan]
            results.append({
                'time': t - n_scan,
                'id': track.track_id,
                'rx': delayed_state[0],
                'ry': delayed_state[1],
                'rz': delayed_state[2],
                'vx': delayed_state[3],
                'vy': delayed_state[4],
                'vz': delayed_state[5] })
    last_t = df['time'].max()
    for track in tracker.active_tracks.values():
        history_len = len(track.best_hypothesis.history_states)
        states_to_flush = min(history_len, n_scan - 1)
        for i in range(states_to_flush, 0, -1):
            delayed_state = track.best_hypothesis.history_states[-i]
            results.append({
                'time': (last_t - i) + 1,
                'id': track.track_id,
                'rx': delayed_state[0],
                'ry': delayed_state[1],
                'rz': delayed_state[2],
                'vx': delayed_state[3],
                'vy': delayed_state[4],
                'vz': delayed_state[5] })
    return pd.DataFrame(results)
