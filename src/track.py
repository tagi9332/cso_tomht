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
    """A single hypothesis for a track with N-Scan history tracking."""
    
    hyp_id: int  # Unique ID for tracking lineage
    state: np.ndarray
    covariance: np.ndarray
    log_score: float = 0.0
    meas_index: Optional[int] = None
    
    # Store the lineage so we can trace back N steps
    history_states: List[np.ndarray] = field(default_factory=list)
    history_ids: List[int] = field(default_factory=list)


class Track:
    def __init__(
        self,
        track_id: int,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        max_hypotheses: int = 10,
        miss_log_likelihood: float = -2.0,
        n_scan_window: int = 5 
    ) -> None:
        self.track_id = track_id
        self.max_hypotheses = max_hypotheses
        self.miss_log_likelihood = miss_log_likelihood
        self.n_scan_window = n_scan_window
        self.age: int = 0
        self.consecutive_misses: int = 0
        self.start_pos = np.asarray(initial_state[0:2], dtype=float)
        
        self._next_hyp_id = 0 # Track-local counter for hypothesis IDs

        # Initialize the root hypothesis
        init_id = self._get_next_id()
        init_state = np.asarray(initial_state, dtype=float)

        self.hypotheses: List[Hypothesis] = [
            Hypothesis(
                hyp_id=init_id, 
                state=init_state,
                covariance=np.asarray(initial_covariance, dtype=float),
                history_states=[init_state], 
                history_ids=[init_id]        
            )
        ]

    def _get_next_id(self) -> int:
        """Helper to generate unique hypothesis IDs within this track."""
        hyp_id = self._next_hyp_id
        self._next_hyp_id += 1
        return hyp_id

    @property
    def best_hypothesis(self) -> Hypothesis:
        return max(self.hypotheses, key=lambda h: h.log_score)

    @property
    def best_state(self) -> np.ndarray:
        return self.best_hypothesis.state

    def expand_hypotheses(
        self,
        predicted_hypotheses: List[Tuple[np.ndarray, np.ndarray]],
        measurement_updates: List[List[Tuple[Optional[int], np.ndarray, np.ndarray, float]]],
    ) -> None:
        new_hypotheses: List[Hypothesis] = []
        for hyp, (x_pred, P_pred), updates in zip(
            self.hypotheses, predicted_hypotheses, measurement_updates
        ):
            for meas_idx, x_upd, P_upd, log_ll in updates:
                new_hypotheses.append(
                    Hypothesis(
                        hyp_id=self._get_next_id(),
                        state=x_upd,
                        covariance=P_upd,
                        log_score=hyp.log_score + log_ll,
                        meas_index=meas_idx,
                        # Pass down the lineage + the parent's state
                        history_states=hyp.history_states + [hyp.state],
                        history_ids=hyp.history_ids + [hyp.hyp_id]
                    )
                )
        self.hypotheses = new_hypotheses
        self._prune()

    def apply_n_scan_pruning(self) -> None:
        """
        N-Scan Pruning: Look N frames into the past of the BEST hypothesis.
        Force all competing realities to collapse to that ancestor.
        """
        if self.age < self.n_scan_window:
            return # Too young to prune N frames back
            
        best_hyp = self.best_hypothesis
        if len(best_hyp.history_ids) < self.n_scan_window:
            return

        # Find the ID of the ancestor from N frames ago
        ancestor_id = best_hyp.history_ids[-self.n_scan_window]

        # Prune: Keep ONLY hypotheses that descend from this ancestor
        survivors = []
        for h in self.hypotheses:
            if len(h.history_ids) >= self.n_scan_window and h.history_ids[-self.n_scan_window] == ancestor_id:
                survivors.append(h)

        self.hypotheses = survivors

    def _prune(self) -> None:
        if len(self.hypotheses) > self.max_hypotheses:
            self.hypotheses = sorted(
                self.hypotheses, key=lambda h: h.log_score, reverse=True
            )[: self.max_hypotheses]

    def normalise_scores(self) -> None:
        best = max(h.log_score for h in self.hypotheses)
        for h in self.hypotheses:
            h.log_score -= best