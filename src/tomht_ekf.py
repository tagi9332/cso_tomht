import numpy as np
from typing import List, Dict, Tuple

from utils.kdtree_association import KDTreeAssociation
from utils.ekf import ExtendedKalmanFilterCWHPinhole 
from src.track import Track
from utils.config_loader import TrackerConfig

class TOMHTTracker:
    """
    Track-Oriented Multiple Hypothesis Tracker (TOMHT) - CWH Pinhole EKF Edition.
    """
    def __init__(self, config: TrackerConfig):
        self.config = config
        
        # Instantiate the 6D CWH Pinhole EKF
        # Removed pixel_pitch from here; the EKF only deals in meters!
        self.kf = ExtendedKalmanFilterCWHPinhole(
            mean_motion=config.mean_motion,
            focal_length=config.focal_length,
            camera_offset_x=config.camera_offset_x,
            dt=config.dt, 
            process_noise_std=config.kf_process_noise, 
            uv_noise_std=config.uv_noise_std,
            range_noise_std=config.range_noise_std
        )
        
        # Note: gate_distance should now be in physical focal plane meters (e.g., 1e-5)
        self.assoc = KDTreeAssociation(
            gate_distance=config.gate_distance, 
            merge_clusters=config.merge_clusters
        )
        
        self.active_tracks: Dict[int, Track] = {}
        self.next_track_id = 0

    def step(self, measurements: np.ndarray) -> Tuple[List[Track], dict]:
        """Process a single frame of measurements [u, v, rho]."""
        measurements = np.atleast_2d(measurements) if len(measurements) > 0 else np.array([])
        
        # --- NEW: Initialize diagnostics dictionary ---
        frame_diag = {
            "gating": [],
            "updates": []
        }
        
        predicted_hyps_map = {} 
        best_predicted_measurements_uv = []
        track_list = list(self.active_tracks.values())
        
        for track in track_list:
            track.age += 1
            preds = []
            for hyp in track.hypotheses:
                x_p, P_p = self.kf.predict(hyp.state, hyp.covariance)
                preds.append((x_p, P_p))
            predicted_hyps_map[track.track_id] = preds
            
            # Project best state to measurement space
            z_pred, _ = self.kf._measurement_model(track.best_state)
            
            # Extract ONLY u and v for KD-Tree clustering to avoid scale imbalance with rho
            best_predicted_measurements_uv.append(z_pred[0:2]) 
            
        best_predicted_measurements_uv = np.array(best_predicted_measurements_uv)

        # Cluster using ONLY the (u, v) focal plane coordinates
        if len(best_predicted_measurements_uv) > 0 and len(measurements) > 0:
            clusters = self.assoc.cluster(best_predicted_measurements_uv, measurements[:, 0:2])
        elif len(measurements) > 0:
            clusters = [{"track_indices": [], "meas_indices": list(range(len(measurements)))}]
        else:
            clusters = [{"track_indices": list(range(len(track_list))), "meas_indices": []}]

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
                    
                    # Option 1: Missed Detection
                    hyp_updates.append((None, x_pred, P_pred, track.miss_log_likelihood))
                    
                    # Option 2: Associate with a gated measurement
                    for m_idx in m_idxs:
                        z = measurements[m_idx] # z is [u, v, rho]
                        dist = self.kf.mahalanobis_distance(x_pred, P_pred, z)
                        gated_in = bool(dist < self.config.mahalanobis_thresh)
                        
                        # --- NEW: Record gating diagnostics ---
                        frame_diag["gating"].append({
                            "track_id": track.track_id,
                            "meas_idx": m_idx,
                            "z_true": z.copy(),
                            "mahalanobis_dist": dist,
                            "gated_in": gated_in
                        })
                        
                        if gated_in: 
                            # --- CRITICAL CHANGE: Catch the 4th return value (kf_diag) ---
                            x_upd, P_upd, log_ll, kf_diag = self.kf.update(x_pred, P_pred, z)
                            hyp_updates.append((m_idx, x_upd, P_upd, log_ll))
                            
                            # --- NEW: Record Deep EKF Metrics ---
                            frame_diag["updates"].append({
                                "track_id": track.track_id,
                                "meas_idx": m_idx,
                                "state_prior": x_pred.copy(),
                                "cov_prior": P_pred.copy(),
                                "state_updated": x_upd.copy(),
                                "cov_updated": P_upd.copy(),
                                "residual": kf_diag["residual"],
                                "kalman_gain": kf_diag["kalman_gain"],
                                "innovation_cov": kf_diag["innovation_cov"],
                                "log_likelihood": log_ll
                            })
                            
                    measurement_updates.append(hyp_updates)
                
                # Apply the branch-and-prune step
                track.expand_hypotheses(predicted_hyps, measurement_updates)
                track.normalise_scores()
                track.apply_n_scan_pruning()
                
                if track.best_hypothesis.meas_index is None:
                    track.consecutive_misses += 1
                else:
                    track.consecutive_misses = 0

        # Prune dead and stationary tracks
        dead_track_ids = []
        
        for t_id, t in self.active_tracks.items():
            if t.consecutive_misses >= self.config.max_misses:
                dead_track_ids.append(t_id)
                continue

            # Check 2D Focal Plane movement (u, v)
            if t.age >= self.config.min_age_to_check:
                # Get current predicted (u, v) from the measurement model
                z_pred, _ = self.kf._measurement_model(t.best_state)
                current_uv = z_pred[0:2]
                
                # Calculate displacement in physical focal plane meters
                uv_displacement_meters = np.linalg.norm(current_uv - t.start_uv)
                
                # Convert to pixels to check against your threshold
                pixel_displacement = uv_displacement_meters / self.config.pixel_pitch
                
                if pixel_displacement < self.config.min_distance_px:
                    dead_track_ids.append(t_id)

        for t_id in dead_track_ids:
            del self.active_tracks[t_id]
            
        # --- CRITICAL CHANGE: Return the tuple ---
        return list(self.active_tracks.values()), frame_diag

    def _initiate_track(self, z: np.ndarray):
        """Helper to spawn a brand new track using EKF initialization."""
        x0, P0 = self.kf.initialize(z)
        
        new_track = Track(
            track_id=self.next_track_id,
            initial_state=x0,
            initial_covariance=P0,
            max_hypotheses=self.config.max_hypotheses,
            miss_log_likelihood=self.config.miss_log_likelihood,
            n_scan_window=self.config.n_scan_window
        )
        # --- FIX: Save the initial (u, v) measurement instead of 3D state ---
        new_track.start_uv = z[0:2].copy() 
        
        self.active_tracks[self.next_track_id] = new_track
        self.next_track_id += 1