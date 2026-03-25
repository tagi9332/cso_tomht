"""Track and Hypothesis data structures for TOMHT.

Each :class:`Track` maintains an ordered list of :class:`Hypothesis`
objects that represent alternative association histories.  During each
scan the tracker generates new hypotheses by pairing the track's
predicted state with candidate measurements (or a missed-detection
null hypothesis).  Low-score hypotheses are pruned to keep the
computational cost manageable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Hypothesis:
    """A single hypothesis for a track.

    Attributes
    ----------
    state : ndarray, shape (4,)
        State estimate [px, py, vx, vy] for this hypothesis.
    covariance : ndarray, shape (4, 4)
        State covariance matrix.
    log_score : float
        Cumulative log-likelihood of this hypothesis.
    meas_index : int or None
        Index of the measurement that was associated at the most recent
        scan (``None`` indicates a missed detection).
    """

    state: np.ndarray
    covariance: np.ndarray
    log_score: float = 0.0
    meas_index: Optional[int] = None


class Track:
    """A single target track with multiple competing hypotheses.

    Parameters
    ----------
    track_id : int
        Unique identifier for this track.
    initial_state : ndarray, shape (4,)
        Initial state estimate [px, py, vx, vy].
    initial_covariance : ndarray, shape (4, 4)
        Initial state covariance.
    max_hypotheses : int, optional
        Maximum number of hypotheses to retain after pruning (default 10).
    miss_log_likelihood : float, optional
        Log-likelihood assigned to a missed-detection hypothesis
        (default -2.0).
    """

    def __init__(
        self,
        track_id: int,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        max_hypotheses: int = 10,
        miss_log_likelihood: float = -2.0,
    ) -> None:
        self.track_id = track_id
        self.max_hypotheses = max_hypotheses
        self.miss_log_likelihood = miss_log_likelihood
        self.age: int = 0
        self.consecutive_misses: int = 0

        self.hypotheses: List[Hypothesis] = [
            Hypothesis(
                state=np.asarray(initial_state, dtype=float),
                covariance=np.asarray(initial_covariance, dtype=float),
            )
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_hypothesis(self) -> Hypothesis:
        """Return the hypothesis with the highest log_score."""
        return max(self.hypotheses, key=lambda h: h.log_score)

    @property
    def best_state(self) -> np.ndarray:
        """Convenience accessor for the best-hypothesis state."""
        return self.best_hypothesis.state

    @property
    def best_covariance(self) -> np.ndarray:
        """Convenience accessor for the best-hypothesis covariance."""
        return self.best_hypothesis.covariance

    # ------------------------------------------------------------------
    # Hypothesis management
    # ------------------------------------------------------------------

    def expand_hypotheses(
        self,
        predicted_hypotheses: List[Tuple[np.ndarray, np.ndarray]],
        measurement_updates: List[List[Tuple[Optional[int], np.ndarray, np.ndarray, float]]],
    ) -> None:
        """Replace current hypotheses with the expanded hypothesis set.

        Parameters
        ----------
        predicted_hypotheses : list of (x_pred, P_pred)
            One entry per current hypothesis.
        measurement_updates : list of lists
            ``measurement_updates[i]`` is a list of
            ``(meas_index, x_upd, P_upd, log_ll)`` tuples representing
            all viable measurement updates (plus missed-detection) for
            hypothesis *i*.
        """
        new_hypotheses: List[Hypothesis] = []
        for hyp, (x_pred, P_pred), updates in zip(
            self.hypotheses, predicted_hypotheses, measurement_updates
        ):
            for meas_idx, x_upd, P_upd, log_ll in updates:
                new_hypotheses.append(
                    Hypothesis(
                        state=x_upd,
                        covariance=P_upd,
                        log_score=hyp.log_score + log_ll,
                        meas_index=meas_idx,
                    )
                )
        self.hypotheses = new_hypotheses
        self._prune()

    def _prune(self) -> None:
        """Keep only the top *max_hypotheses* hypotheses by log_score."""
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses = sorted(
                self.hypotheses, key=lambda h: h.log_score, reverse=True
            )[: self.max_hypotheses]

    def normalise_scores(self) -> None:
        """Shift log_scores so that the best hypothesis has score 0.

        This prevents floating-point underflow over long trajectories.
        """
        best = max(h.log_score for h in self.hypotheses)
        for h in self.hypotheses:
            h.log_score -= best
