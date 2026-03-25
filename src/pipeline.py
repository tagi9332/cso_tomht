"""Track-Oriented Multi-Hypothesis Tracking (TOMHT) pipeline.

The pipeline processes a stream of measurement scans and maintains a set
of :class:`~src.track.Track` objects.  For each new scan it:

1. **Predicts** all tracks one time step ahead using the
   :class:`~utils.kalman_filter.KalmanFilter2D`.
2. **Gates** measurements to predicted positions using
   :class:`~utils.kdtree_association.KDTreeAssociation`.
3. **Expands hypotheses** – for each track, one new hypothesis is created
   per gated measurement plus a missed-detection hypothesis.
4. **Scores** each new hypothesis with the Kalman-filter log-likelihood.
5. **Prunes** hypotheses beyond *max_hypotheses* per track.
6. **Initiates** new tentative tracks from un-gated measurements.
7. **Deletes** tracks that have exceeded the maximum allowed number of
   consecutive missed detections.

Usage
-----
>>> from src.pipeline import TOMHTPipeline
>>> pipeline = TOMHTPipeline(gate_distance=15.0, dt=1.0)
>>> for scan in measurement_scans:
...     tracks = pipeline.process(scan)
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.track import Hypothesis, Track
from utils.kalman_filter import KalmanFilter2D
from utils.kdtree_association import KDTreeAssociation


class TOMHTPipeline:
    """Track-Oriented Multi-Hypothesis Tracking pipeline.

    Parameters
    ----------
    gate_distance : float
        Euclidean gating radius used by the k-d tree association step.
    dt : float
        Time step between consecutive scans (seconds).
    process_noise_std : float
        Process noise standard deviation passed to
        :class:`~utils.kalman_filter.KalmanFilter2D`.
    measurement_noise_std : float
        Measurement noise standard deviation passed to
        :class:`~utils.kalman_filter.KalmanFilter2D`.
    max_hypotheses : int
        Maximum hypotheses retained per track.
    max_missed : int
        A track is deleted after this many consecutive missed detections.
    min_hits_confirm : int
        Minimum number of scans (age) before a track is considered
        *confirmed* (returned by :meth:`confirmed_tracks`).
    miss_log_likelihood : float
        Log-likelihood penalty assigned to a missed-detection hypothesis.
    """

    def __init__(
        self,
        gate_distance: float = 10.0,
        dt: float = 1.0,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 1.0,
        max_hypotheses: int = 10,
        max_missed: int = 3,
        min_hits_confirm: int = 2,
        miss_log_likelihood: float = -2.0,
    ) -> None:
        self.kf = KalmanFilter2D(
            dt=dt,
            process_noise_std=process_noise_std,
            measurement_noise_std=measurement_noise_std,
        )
        self.assoc = KDTreeAssociation(gate_distance=gate_distance)
        self.max_missed = max_missed
        self.min_hits_confirm = min_hits_confirm
        self.miss_log_likelihood = miss_log_likelihood
        self.max_hypotheses = max_hypotheses

        self._tracks: List[Track] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> List[Track]:
        """All currently maintained tracks (confirmed + tentative)."""
        return list(self._tracks)

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Tracks that have been active for at least *min_hits_confirm* scans."""
        return [t for t in self._tracks if t.age >= self.min_hits_confirm]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, measurements: np.ndarray) -> List[Track]:
        """Process a single scan of measurements.

        Parameters
        ----------
        measurements : array_like, shape (M, 2)
            The 2-D [x, y] measurements received at this time step.
            Pass an empty array (shape ``(0, 2)``) for a scan with no
            detections.

        Returns
        -------
        tracks : list of Track
            The updated list of all active tracks after processing this
            scan.
        """
        measurements = np.atleast_2d(np.asarray(measurements, dtype=float))
        if measurements.shape == (1, 2) and np.all(measurements == 0) and len(measurements) == 1:
            # Allow passing an empty list naturally
            pass
        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, 2)

        # Step 1 – predict all track hypotheses
        predicted: Dict[int, List[tuple]] = {}  # track_id -> [(x_pred, P_pred), ...]
        predicted_best: List[np.ndarray] = []   # best predicted pos per track

        for track in self._tracks:
            preds = []
            for hyp in track.hypotheses:
                x_pred, P_pred = self.kf.predict(hyp.state, hyp.covariance)
                preds.append((x_pred, P_pred))
            predicted[track.track_id] = preds
            # Use best-hypothesis predicted position for gating
            best_idx = np.argmax([h.log_score for h in track.hypotheses])
            predicted_best.append(preds[best_idx][0][:2])

        # Step 2 – gate measurements to tracks using k-d tree
        if len(self._tracks) > 0 and len(measurements) > 0:
            association = self.assoc.associate(
                np.array(predicted_best), measurements
            )
        else:
            association = {}

        # Step 3 & 4 – expand and score hypotheses; update tracks
        gated_meas: set = set()
        for i, track in enumerate(self._tracks):
            preds = predicted[track.track_id]
            gated_indices = association.get(i, [])

            measurement_updates_per_hyp = []
            for x_pred, P_pred in preds:
                updates = []
                # Measurement-associated hypotheses
                for m_idx in gated_indices:
                    z = measurements[m_idx]
                    x_upd, P_upd, _ = self.kf.update(x_pred, P_pred, z)
                    # Score using -0.5 * Mahalanobis² (exponent only) so that
                    # the score is independent of |S| and comparable to the
                    # fixed miss_log_likelihood threshold.
                    d = self.kf.mahalanobis_distance(x_pred, P_pred, z)
                    score = -0.5 * d * d
                    updates.append((m_idx, x_upd, P_upd, score))

                # Missed-detection hypothesis
                updates.append(
                    (None, x_pred, P_pred, self.miss_log_likelihood)
                )
                measurement_updates_per_hyp.append(updates)

            track.expand_hypotheses(preds, measurement_updates_per_hyp)
            track.normalise_scores()
            track.age += 1

            if track.best_hypothesis.meas_index is None:
                track.consecutive_misses += 1
            else:
                track.consecutive_misses = 0
                gated_meas.add(track.best_hypothesis.meas_index)

        # Step 5 – delete tracks that missed too many times
        self._tracks = [
            t for t in self._tracks if t.consecutive_misses <= self.max_missed
        ]

        # Step 6 – initiate new tentative tracks from un-gated measurements
        # Collect all measurements gated to *any* track (not just best)
        all_gated = {m for idxs in association.values() for m in idxs}
        for m_idx in range(len(measurements)):
            if m_idx not in all_gated:
                z = measurements[m_idx]
                x0, P0 = self.kf.initialize(z[:2])
                new_track = Track(
                    track_id=self._next_id,
                    initial_state=x0,
                    initial_covariance=P0,
                    max_hypotheses=self.max_hypotheses,
                    miss_log_likelihood=self.miss_log_likelihood,
                )
                self._tracks.append(new_track)
                self._next_id += 1

        return list(self._tracks)

    def reset(self) -> None:
        """Clear all tracks and reset the track-ID counter."""
        self._tracks = []
        self._next_id = 0
