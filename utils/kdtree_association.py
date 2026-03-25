"""KD-Tree Data Association for MHT.

Provides spatial gating and clustering of measurements to predicted track positions.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
from scipy.spatial import KDTree

class KDTreeAssociation:
    """Associates measurements to tracks using a k-d tree for Euclidean gating.
    
    Attributes:
        gate_distance: Max Euclidean distance for valid association.
        merge_clusters: If True, merges tracks sharing measurements into joint clusters.
    """

    def __init__(self, gate_distance: float, merge_clusters: bool = True) -> None:
        if gate_distance <= 0:
            raise ValueError("gate_distance must be positive.")
        self.gate_distance = gate_distance
        self.merge_clusters = merge_clusters

    def associate(self, predicted_positions: np.ndarray, measurements: np.ndarray) -> Dict[int, List[int]]:
        """Maps track indices to measurement indices within gate_distance."""
        preds = np.atleast_2d(np.asarray(predicted_positions, dtype=float))
        meas = np.atleast_2d(np.asarray(measurements, dtype=float))

        if len(meas) == 0:
            return {}

        tree = KDTree(meas)
        association: Dict[int, List[int]] = {}

        for idx, pos in enumerate(preds):
            gated = tree.query_ball_point(pos, self.gate_distance)
            if gated:
                association[idx] = sorted(gated)

        return association

    def cluster(self, predicted_positions: np.ndarray, measurements: np.ndarray) -> List[Dict[str, List[int]]]:
        """Partitions tracks and measurements into independent groups for the MHT solver.
        
        Returns:
            List of dicts containing 'track_indices' and 'meas_indices'. 
            Empty 'track_indices' indicate potential new track initiations.
        """
        assoc = self.associate(predicted_positions, measurements)
        n_meas = len(np.atleast_2d(measurements))

        if not self.merge_clusters:
            gated_m = {m for midx in assoc.values() for m in midx}
            clusters = [{"track_indices": [t], "meas_indices": m} for t, m in assoc.items()]
            clusters += [{"track_indices": [], "meas_indices": [m]} for m in range(n_meas) if m not in gated_m]
            return clusters

        # Union-Find to merge overlapping gates
        parent = list(range(len(np.atleast_2d(predicted_positions))))

        def _find(i: int) -> int:
            if parent[i] == i: return i
            parent[i] = _find(parent[i])
            return parent[i]

        def _union(i: int, j: int):
            root_i, root_j = _find(i), _find(j)
            if root_i != root_j: parent[root_i] = root_j

        # Link tracks that share a measurement
        m_to_t: Dict[int, List[int]] = {}
        for t, midx in assoc.items():
            for m in midx:
                m_to_t.setdefault(m, []).append(t)

        for tracks in m_to_t.values():
            for k in range(1, len(tracks)):
                _union(tracks[0], tracks[k])

        # Group by component
        groups: Dict[int, Dict[str, set]] = {}
        for t, midx in assoc.items():
            root = _find(t)
            if root not in groups:
                groups[root] = {"track_indices": set(), "meas_indices": set()}
            groups[root]["track_indices"].add(t)
            groups[root]["meas_indices"].update(midx)

        # Build output list
        clusters = [
            {"track_indices": sorted(v["track_indices"]), "meas_indices": sorted(v["meas_indices"])}
            for v in groups.values()
        ]

        # Add unassociated measurements
        gated_m = {m for v in groups.values() for m in v["meas_indices"]}
        clusters += [{"track_indices": [], "meas_indices": [m]} for m in range(n_meas) if m not in gated_m]

        return clusters