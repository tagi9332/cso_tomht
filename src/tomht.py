import numpy as np
from typing import List, Dict

from utils.kdtree_association import KDTreeAssociation
from utils.kalman_filter import KalmanFilter2D
from src.track import Track
from utils.config_loader import TrackerConfig

class TOMHTTracker:
    """
    Track-Oriented Multiple Hypothesis Tracker (TOMHT).

    This tracker maintains multiple parallel association hypotheses for each 
    active track to handle measurement ambiguity (clutter, missed detections). 
    It relies on a Kalman Filter for state estimation, KD-Tree for spatial 
    clustering, and N-Scan pruning to manage exponential branch growth.

    Parameters
    ----------
    dt : float, optional
        Time step between consecutive frames, by default 1.0.
    gate_distance : float, optional
        Maximum spatial distance for initial measurement clustering and gating.
    max_misses : int, optional
        Maximum number of consecutive missed detections before a track is 
        considered dead and pruned.
    max_hypotheses : int, optional
        The maximum number of distinct hypothesis branches to maintain per 
        track.
    n_scan_window : int, optional
        The depth of the history window used for N-Scan pruning. Competing 
        hypotheses are forced to collapse to the ancestor from `N` frames ago.
    """
    def __init__(self, config: TrackerConfig):
        self.config = config
        
        # Intantiate 2D Kalman Filter and KD-Tree Association modules
        self.kf = KalmanFilter2D(
            dt=config.dt, 
            process_noise_std=config.kf_process_noise, 
            measurement_noise_std=config.kf_meas_noise
        )
        self.assoc = KDTreeAssociation(
            gate_distance=config.gate_distance, 
            merge_clusters=config.merge_clusters
        )
        
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0

    def step(self, measurements: np.ndarray) -> List[Track]:
        """Process a single frame of measurements."""
        measurements = np.atleast_2d(measurements) if len(measurements) > 0 else np.array([])
        
        # Predict next state for all existing hypotheses in active tracks
        predicted_hyps_map = {} 
        best_positions = []
        track_list = list(self.active_tracks.values())
        
        for track in track_list:
            track.age += 1
            preds = []
            for hyp in track.hypotheses:
                x_p, P_p = self.kf.predict(hyp.state, hyp.covariance)
                preds.append((x_p, P_p))
            predicted_hyps_map[track.track_id] = preds
            
            best_positions.append(track.best_state[0:2]) 
            
        best_positions = np.array(best_positions)

        # Cluster measurements with predicted positions using KD-Tree
        if len(best_positions) > 0 and len(measurements) > 0:
            clusters = self.assoc.cluster(best_positions, measurements)
        elif len(measurements) > 0:
            clusters = [{"track_indices": [], "meas_indices": list(range(len(measurements)))}]
        else:
            clusters = [{"track_indices": list(range(len(track_list))), "meas_indices": []}]

        # For each cluster, generate new hypotheses for associated tracks and measurements
        for cluster in clusters:
            t_idxs = cluster["track_indices"]
            m_idxs = cluster["meas_indices"]
            
            # Case A: Unassociated measurements -> New Tracks
            if len(t_idxs) == 0:
                for m_idx in m_idxs:
                    self._initiate_track(measurements[m_idx])
                continue
                
            # Case B: Update existing tracks in the cluster
            for t_idx in t_idxs:
                track = track_list[t_idx]
                predicted_hyps = predicted_hyps_map[track.track_id]
                measurement_updates = []
                
                for x_pred, P_pred in predicted_hyps:
                    hyp_updates = []
                    
                    # Option 1: Missed Detection (Null Hypothesis)
                    hyp_updates.append((None, x_pred, P_pred, track.miss_log_likelihood))
                    
                    # Option 2: Associate with a gated measurement
                    for m_idx in m_idxs:
                        z = measurements[m_idx]
                        dist = self.kf.mahalanobis_distance(x_pred, P_pred, z)
                        
                        # Use Mahalanobis threshold
                        if dist < self.config.mahalanobis_thresh: 
                            x_upd, P_upd, log_ll = self.kf.update(x_pred, P_pred, z)
                            hyp_updates.append((m_idx, x_upd, P_upd, log_ll))
                            
                    measurement_updates.append(hyp_updates)
                
                # Apply the branch-and-prune step
                track.expand_hypotheses(predicted_hyps, measurement_updates)
                track.normalise_scores()
                
                # Apply N-Scan Pruning
                track.apply_n_scan_pruning()
                
                # Update track management stats
                if track.best_hypothesis.meas_index is None:
                    track.consecutive_misses += 1
                else:
                    track.consecutive_misses = 0

        # Prune dead and stationary tracks
        dead_track_ids = []
        
        for t_id, t in self.active_tracks.items():

            # Prune tracks that have too many consecutive misses
            if t.consecutive_misses >= self.config.max_misses:
                dead_track_ids.append(t_id)
                continue

            # Prune tracks that haven't moved significantly from their start position after a certain age    
            if t.age >= self.config.min_age_to_check:
                current_pos = t.best_state[0:2]
                displacement = np.linalg.norm(current_pos - t.start_pos)
                
                # If the track hasn't moved enough, mark it for deletion
                if displacement < self.config.min_distance_px:
                    dead_track_ids.append(t_id)

        # Delete the marked tracks
        for t_id in dead_track_ids:
            del self.active_tracks[t_id]
            
        return list(self.active_tracks.values())

    def _initiate_track(self, z: np.ndarray):
        """Helper to spawn a brand new track."""
        x0, P0 = self.kf.initialize(z)
        new_track = Track(
            track_id=self.next_track_id,
            initial_state=x0,
            initial_covariance=P0,
            max_hypotheses=self.config.max_hypotheses,
            miss_log_likelihood=self.config.miss_log_likelihood,
            n_scan_window=self.config.n_scan_window
        )
        self.active_tracks[self.next_track_id] = new_track
        self.next_track_id += 1