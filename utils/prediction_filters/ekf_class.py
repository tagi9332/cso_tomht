import numpy as np
from typing import Tuple
from utils.config_loader import EKFOptions
from utils.prediction_filters.compute_q_discrete import compute_q_discrete 

class CWH_AnglesOnly_EKF:
    def __init__(self, dt: float, ekf_options: EKFOptions, R_matrix: np.ndarray):
        self.dt = dt
        self.options = ekf_options
        self.R = R_matrix
        self.n_states = 6
        self.I = np.eye(self.n_states)
        
        # Pull coefficients for easy access
        self.n_mean_motion = ekf_options.coeffs.mean_motion
        self.f = ekf_options.coeffs.focal_length
        self.D = ekf_options.coeffs.nominal_distance

    def compute_cwh_stm(self, dt: float) -> np.ndarray:
        """Computes the analytical Clohessy-Wiltshire-Hill State Transition Matrix."""
        n = self.n_mean_motion
        nt = n * dt
        sn, cs = np.sin(nt), np.cos(nt)
        
        Phi = np.zeros((6, 6))
        # Pos to Pos
        Phi[0, 0], Phi[0, 3], Phi[0, 4] = 4 - 3*cs, (1/n)*sn, (2/n)*(1-cs)
        Phi[1, 0], Phi[1, 1], Phi[1, 3], Phi[1, 4] = 6*(sn-nt), 1, (2/n)*(cs-1), (1/n)*(4*sn-3*nt)
        Phi[2, 2], Phi[2, 5] = cs, (1/n)*sn
        # Vel to Pos/Vel
        Phi[3, 0], Phi[3, 3], Phi[3, 4] = 3*n*sn, cs, 2*sn
        Phi[4, 0], Phi[4, 3], Phi[4, 4] = 6*n*(cs-1), -2*sn, 4*cs-3
        Phi[5, 2], Phi[5, 5] = -n*sn, cs
        return Phi

    def compute_h_nonlinear(self, state: np.ndarray) -> np.ndarray:
        """Projects 6D RIC state to 2D focal plane + 1D pseudo-range."""
        x, y, z = state[0], state[1], state[2]
        u = self.f * (y / (self.D + x))
        v = self.f * (z / (self.D + x))
        return np.array([u, v, x]) # [u, v, x_pseudo]

    def compute_H_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Jacobian of measurement model evaluated at current state."""
        x, y, z = state[0], state[1], state[2]
        denom = self.D + x
        
        H = np.zeros((3, self.n_states))
        H[0, 0] = -self.f * y / (denom**2)
        H[0, 1] = self.f / denom
        H[1, 0] = -self.f * z / (denom**2)
        H[1, 2] = self.f / denom
        H[2, 0] = 1.0 
        return H

    def predict(self, x: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates state and covariance to the next timestep."""
        Phi = self.compute_cwh_stm(self.dt)
        Q_k = compute_q_discrete(self.dt, x, self.options)
        
        x_pred = Phi @ x
        P_pred = Phi @ P @ Phi.T + Q_k
        return x_pred, P_pred

    def update(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Performs EKF measurement update and returns log-likelihood."""
        z_pred = self.compute_h_nonlinear(x_pred)
        y = z - z_pred  # Innovation
        
        H = self.compute_H_jacobian(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Update state and covariance (Joseph Form for stability)
        x_post = x_pred + K @ y
        IKH = self.I - K @ H
        P_post = IKH @ P_pred @ IKH.T + K @ self.R @ K.T
        
        # Log-Likelihood for TOMHT scoring
        dim = len(z)
        log_ll = -0.5 * (y.T @ np.linalg.solve(S, y) + np.log(np.linalg.det(S)) + dim * np.log(2 * np.pi))
        
        return x_post, P_post, log_ll

    def mahalanobis_distance(self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray) -> float:
        """Computes distance for gating."""
        z_pred = self.compute_h_nonlinear(x_pred)
        y = z - z_pred
        H = self.compute_H_jacobian(x_pred)
        S = H @ P_pred @ H.T + self.R
        return np.sqrt(y.T @ np.linalg.solve(S, y))

    def initialize(self, z_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Initializes a 6D state from a 3D measurement [u, v, x_pseudo]."""
        u, v, x_p = z_3d
        # Inverse of the measurement model to get y and z
        # y = u * (D + x) / f
        y_init = u * (self.D + x_p) / self.f
        z_init = v * (self.D + x_p) / self.f
        
        x0 = np.array([x_p, y_init, z_init, 0.0, 0.0, 0.0]) # Zero initial velocity
        P0 = np.diag([1e4, 2.5e5, 2.5e5, 1e4, 1e6, 1e6])
        return x0, P0