import json
import re
from dataclasses import dataclass
from typing import Any, Dict

def _load_jsonx(filepath: str) -> Dict[str, Any]:
    """Reads a JSONX file, strips // comments, and parses it."""
    with open(filepath, 'r') as f:
        raw_text = f.read()
    
    clean_text = re.sub(r'//.*', '', raw_text)
    
    return json.loads(clean_text)

@dataclass
class TrackerConfig:
    """Data container for TOMHT parameters."""
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
    

@dataclass
class EKFTrackerConfig:
    """Unified configuration for the CWH-EKF TOMHT tracking pipeline."""
    
    # ==========================================
    # Optical & Sensor Parameters
    # ==========================================
    focal_length: float = 4.0
    camera_offset_x: float = 0.0
    uv_noise_std: float = 7.5e-5
    pixel_pitch: float = 1.5e-6  
    
    # ==========================================
    # Orbital & Physics Parameters
    # ==========================================
    mean_motion: float = 7.292115e-5  
    dt: float = 3600.0                   
    range_noise_std: float = 5.0      
    
    # ==========================================
    # Kalman Filter Tuning
    # ==========================================
    kf_process_noise: float = 1e-2    
    mahalanobis_thresh: float = 5  # Default 95% confidence for 4-DOF
    
    # ==========================================
    # Association
    # ==========================================
    gate_distance: float = 100       
    merge_clusters: bool = False
    
    # ==========================================
    # Track Management
    # ==========================================
    max_misses: int = 500
    min_hits_confirm: int = 3
    min_age_to_check: int = 3
    min_distance_px: float = 1      
    
    # ==========================================
    # Hypothesis Management
    # ==========================================
    max_hypotheses: int = 10
    n_scan_window: int = 5            
    miss_log_likelihood: float = -15.0

    @classmethod
    def from_jsonx(cls, filepath: str) -> "EKFTrackerConfig":
        """Loads config from a nested JSON/JSONX file, falling back to defaults if keys are missing."""
        try:
            data = _load_jsonx(filepath)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[!] Warning: Could not load or parse {filepath}. Using default parameters. ({e})")
            return cls()

        # Safely extract nested dictionaries to prevent KeyError if sections are missing
        kf_data = data.get('kalman_filter', {})
        assoc_data = data.get('association', {})
        tm_data = data.get('track_management', {})
        hyp_data = data.get('hypothesis', {})

        return cls(
            # Top level
            dt=data.get('dt', cls.dt),
            mean_motion=data.get('mean_motion', cls.mean_motion),
            range_noise_std=data.get('range_noise_std', cls.range_noise_std),
            focal_length=data.get('focal_length', cls.focal_length),
            camera_offset_x=data.get('camera_offset_x', cls.camera_offset_x),
            uv_noise_std=data.get('uv_noise_std', cls.uv_noise_std),
            
            # Kalman Filter
            # ---> ADDED: kf_process_noise loading <---
            kf_process_noise=kf_data.get('process_noise_std', cls.kf_process_noise),
            mahalanobis_thresh=kf_data.get('mahalanobis_threshold', cls.mahalanobis_thresh),
            
            # Association
            gate_distance=assoc_data.get('gate_distance', cls.gate_distance),
            merge_clusters=assoc_data.get('merge_clusters', cls.merge_clusters),
            
            # Track Management
            max_misses=tm_data.get('max_misses', cls.max_misses),
            min_hits_confirm=tm_data.get('min_hits_confirm', cls.min_hits_confirm),
            min_age_to_check=tm_data.get('min_age_to_check', cls.min_age_to_check),
            min_distance_px=tm_data.get('min_distance_px', cls.min_distance_px),
            
            # Hypothesis
            max_hypotheses=hyp_data.get('max_hypotheses', cls.max_hypotheses),
            n_scan_window=hyp_data.get('n_scan_window', cls.n_scan_window),
            miss_log_likelihood=hyp_data.get('miss_log_likelihood', cls.miss_log_likelihood),
            pixel_pitch=data.get('pixel_pitch', cls.pixel_pitch)

        )