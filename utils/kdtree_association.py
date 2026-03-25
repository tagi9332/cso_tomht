"""K-D Tree Clustering Association utility.

Provides fast nearest-neighbour gating and clustering of measurements to
predicted track positions using a scipy k-d tree.

Typical usage
-------------
>>> assoc = KDTreeAssociation(gate_distance=10.0)
>>> clusters = assoc.associate(predicted_positions, measurements)
>>> for track_idx, meas_indices in clusters.items():
...     print(f"Track {track_idx} gated measurements: {meas_indices}")
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.spatial import KDTree


class KDTreeAssociation:
    """Associate measurements to predicted track positions via a k-d tree.

    A k-d tree is built from the current set of measurements each time
    :meth:`associate` is called.  For every predicted track position the
    tree is queried for all measurements that fall within *gate_distance*
    (Euclidean).  Optionally, connected groups of measurements that share
    at least one common track can be merged into *clusters* so that the
    MHT solver works on them jointly.

    Parameters
    ----------
    gate_distance : float
        Maximum Euclidean distance (in the same units as the measurement
        coordinates) between a predicted position and a measurement for
        the measurement to be considered a valid association candidate.
    merge_clusters : bool, optional
        When ``True`` (default) measurements that are in the gate of
        multiple tracks are merged into a single cluster so that the
        joint hypothesis space is solved together.
    """

    def __init__(self, gate_distance: float, merge_clusters: bool = True) -> None:
        if gate_distance <= 0:
            raise ValueError("gate_distance must be positive.")
        self.gate_distance = gate_distance
        self.merge_clusters = merge_clusters

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def associate(
        self,
        predicted_positions: np.ndarray,
        measurements: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Gate measurements to track predicted positions.

        Parameters
        ----------
        predicted_positions : array_like, shape (N, 2)
            The 2-D [x, y] predicted positions of *N* tracks.
        measurements : array_like, shape (M, 2)
            The 2-D [x, y] measurement positions at the current time step.

        Returns
        -------
        association : dict[int, list[int]]
            Mapping from track index → list of measurement indices that
            fall within *gate_distance* of that track's predicted position.
            Tracks with no gated measurements are **not** included.
        """
        predicted_positions = np.atleast_2d(np.asarray(predicted_positions, dtype=float))
        measurements = np.atleast_2d(np.asarray(measurements, dtype=float))

        if len(measurements) == 0:
            return {}

        tree = KDTree(measurements)
        association: Dict[int, List[int]] = {}

        for track_idx, pos in enumerate(predicted_positions):
            indices = tree.query_ball_point(pos, self.gate_distance)
            if indices:
                association[track_idx] = sorted(indices)

        return association

    def cluster(
        self,
        predicted_positions: np.ndarray,
        measurements: np.ndarray,
    ) -> List[Dict[str, List[int]]]:
        """Partition tracks and measurements into independent clusters.

        Two tracks belong to the same cluster if they share at least one
        common measurement in their gate.  Measurements ungated to any
        track form individual single-measurement clusters (new-track
        candidates).

        Parameters
        ----------
        predicted_positions : array_like, shape (N, 2)
        measurements : array_like, shape (M, 2)

        Returns
        -------
        clusters : list of dict
            Each element is a dict with keys:

            * ``"track_indices"``  – list of track indices in this cluster.
            * ``"meas_indices"``   – list of measurement indices in this
              cluster.

            Clusters where *track_indices* is empty represent un-associated
            measurements (potential new-track initiations).
        """
        association = self.associate(predicted_positions, measurements)

        measurements = np.atleast_2d(np.asarray(measurements, dtype=float))
        n_meas = len(measurements)

        if not self.merge_clusters:
            clusters = [
                {"track_indices": [t], "meas_indices": midx}
                for t, midx in association.items()
            ]
            # Add un-gated measurements as empty-track clusters
            gated_meas = {m for midx in association.values() for m in midx}
            for m in range(n_meas):
                if m not in gated_meas:
                    clusters.append({"track_indices": [], "meas_indices": [m]})
            return clusters

        # Union-Find over track indices to merge overlapping gates
        parent = list(range(len(np.atleast_2d(predicted_positions))))

        def _find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i: int, j: int) -> None:
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        # Merge tracks that share a measurement
        meas_to_tracks: Dict[int, List[int]] = {}
        for t, midx in association.items():
            for m in midx:
                meas_to_tracks.setdefault(m, []).append(t)

        for tracks in meas_to_tracks.values():
            for k in range(1, len(tracks)):
                _union(tracks[0], tracks[k])

        # Group by root
        cluster_map: Dict[int, Dict[str, set]] = {}
        for t, midx in association.items():
            root = _find(t)
            if root not in cluster_map:
                cluster_map[root] = {"track_indices": set(), "meas_indices": set()}
            cluster_map[root]["track_indices"].add(t)
            cluster_map[root]["meas_indices"].update(midx)

        clusters: List[Dict[str, List[int]]] = [
            {
                "track_indices": sorted(v["track_indices"]),
                "meas_indices": sorted(v["meas_indices"]),
            }
            for v in cluster_map.values()
        ]

        # Append un-gated measurements as potential new-track initiations
        gated_meas = {m for v in cluster_map.values() for m in v["meas_indices"]}
        for m in range(n_meas):
            if m not in gated_meas:
                clusters.append({"track_indices": [], "meas_indices": [m]})

        return clusters
