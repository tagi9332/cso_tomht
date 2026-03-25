import json
import re
from dataclasses import dataclass
from typing import Any, Dict

def _load_jsonx(filepath: str) -> Dict[str, Any]:
    """Reads a JSONX file, strips // comments, and parses it."""
    with open(filepath, 'r') as f:
        raw_text = f.read()
    
    # Use regex to strip out single-line comments (// ...)
    clean_text = re.sub(r'//.*', '', raw_text)
    
    return json.loads(clean_text)

@dataclass
class TrackerConfig:
    """Strongly-typed data container for TOMHT parameters."""
    dt: float
    
    # Kalman Filter
    kf_process_noise: float
    kf_meas_noise: float
    mahalanobis_thresh: float
    
    # Association
    gate_distance: float
    merge_clusters: bool
    
    # Track Management
    max_misses: int
    min_hits_confirm: int
    min_age_to_check: int
    min_distance_px: float
    
    # Hypothesis
    max_hypotheses: int
    n_scan_window: int
    miss_log_likelihood: float

    @classmethod
    def from_jsonx(cls, filepath: str) -> "TrackerConfig":
        """Instantiates the config object directly from the JSONX file."""
        data = _load_jsonx(filepath)
        
        return cls(
            dt=data['dt'],
            
            kf_process_noise=data['kalman_filter']['process_noise_std'],
            kf_meas_noise=data['kalman_filter']['measurement_noise_std'],
            mahalanobis_thresh=data['kalman_filter']['mahalanobis_threshold'],
            
            gate_distance=data['association']['gate_distance'],
            merge_clusters=data['association']['merge_clusters'],
            
            max_misses=data['track_management']['max_misses'],
            min_hits_confirm=data['track_management']['min_hits_confirm'],
            min_age_to_check=data['track_management']['min_age_to_check'],
            min_distance_px=data['track_management']['min_distance_px'],
            
            max_hypotheses=data['hypothesis']['max_hypotheses'],
            n_scan_window=data['hypothesis']['n_scan_window'],
            miss_log_likelihood=data['hypothesis']['miss_log_likelihood']
        )