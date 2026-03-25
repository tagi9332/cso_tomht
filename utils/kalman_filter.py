"""2D Linearized Kalman Filter utility.

Models a target moving at constant velocity in the 2-D plane.

State vector  : x = [px, py, vx, vy]^T
Measurement   : z = [px, py]^T  (position only)
State transition (constant-velocity, time step dt):

    F = | 1  0  dt  0 |
        | 0  1   0 dt |
        | 0  0   1  0 |
        | 0  0   0  1 |

Measurement matrix:

    H = | 1  0  0  0 |
        | 0  1  0  0 |
"""

from __future__ import annotations

import numpy as np


class KalmanFilter2D:
    """Linear Kalman filter for 2-D constant-velocity motion.

    Parameters
    ----------
    dt : float
        Time step between consecutive measurements (seconds).
    process_noise_std : float
        Spectral density of the continuous white noise acceleration 
        applied to the target (used to build the process noise covariance Q).
    measurement_noise_std : float
        Standard deviation of the position measurement noise (used to
        build the measurement noise covariance R).
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 1.0,
    ) -> None:
        self.dt = dt

        # State transition matrix
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Measurement matrix (observe position only)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )

        # Process noise covariance (continuous white-noise acceleration model / SNC)
        q = process_noise_std ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        
        self.Q = q * np.array(
            [
                [dt3 / 3.0, 0.0, dt2 / 2.0, 0.0],
                [0.0, dt3 / 3.0, 0.0, dt2 / 2.0],
                [dt2 / 2.0, 0.0, dt, 0.0],
                [0.0, dt2 / 2.0, 0.0, dt],
            ]
        )

        # Measurement noise covariance
        r = measurement_noise_std ** 2
        self.R = r * np.eye(2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(
        self,
        position: np.ndarray,
        velocity: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create an initial state estimate from a position measurement.

        Parameters
        ----------
        position : array_like, shape (2,)
            Initial [px, py] position.
        velocity : array_like, shape (2,), optional
            Initial [vx, vy] velocity estimate.  Defaults to ``[0, 0]``.
        P0 : ndarray, shape (4, 4), optional
            Initial state covariance.  Defaults to a diagonal matrix with
            large values for velocity states.

        Returns
        -------
        x : ndarray, shape (4,)
            State vector  [px, py, vx, vy].
        P : ndarray, shape (4, 4)
            State covariance matrix.
        """
        position = np.asarray(position, dtype=float)
        velocity = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=float)
        x = np.concatenate([position, velocity])

        if P0 is None:
            r = self.R[0, 0]
            P0 = np.diag([r, r, 100.0, 100.0])
        P = np.asarray(P0, dtype=float)
        return x, P

    def predict(
        self, x: np.ndarray, P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward by one time step.

        Parameters
        ----------
        x : ndarray, shape (4,)
            Current state vector.
        P : ndarray, shape (4, 4)
            Current state covariance.

        Returns
        -------
        x_pred : ndarray, shape (4,)
            Predicted state vector.
        P_pred : ndarray, shape (4, 4)
            Predicted state covariance.
        """
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Incorporate a new position measurement.

        Parameters
        ----------
        x_pred : ndarray, shape (4,)
            Predicted state vector (output of :meth:`predict`).
        P_pred : ndarray, shape (4, 4)
            Predicted covariance (output of :meth:`predict`).
        z : array_like, shape (2,)
            Measurement vector [px, py].

        Returns
        -------
        x_upd : ndarray, shape (4,)
            Updated state vector.
        P_upd : ndarray, shape (4, 4)
            Updated state covariance.
        log_likelihood : float
            Log-likelihood of the measurement given the predicted state,
            useful for hypothesis scoring in MHT.
        """
        z = np.asarray(z, dtype=float)
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        innovation = z - self.H @ x_pred
        x_upd = x_pred + K @ innovation
        I = np.eye(len(x_pred))
        P_upd = (I - K @ self.H) @ P_pred

        # Scalar log-likelihood of the Gaussian innovation
        n = len(z)
        log_likelihood = -0.5 * (
            float(innovation @ np.linalg.solve(S, innovation))
            + np.log(np.linalg.det(S))
            + n * np.log(2 * np.pi)
        )
        return x_upd, P_upd, log_likelihood

    def innovation_covariance(self, P_pred: np.ndarray) -> np.ndarray:
        """Return the measurement-space innovation covariance S = H P H^T + R.

        Parameters
        ----------
        P_pred : ndarray, shape (4, 4)

        Returns
        -------
        S : ndarray, shape (2, 2)
        """
        return self.H @ P_pred @ self.H.T + self.R

    def mahalanobis_distance(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute the Mahalanobis distance between a predicted state and a measurement.

        Parameters
        ----------
        x_pred : ndarray, shape (4,)
        P_pred : ndarray, shape (4, 4)
        z : array_like, shape (2,)

        Returns
        -------
        distance : float
        """
        z = np.asarray(z, dtype=float)
        S = self.innovation_covariance(P_pred)
        innovation = z - self.H @ x_pred
        return float(np.sqrt(innovation @ np.linalg.solve(S, innovation)))
