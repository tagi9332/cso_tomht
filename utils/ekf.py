from __future__ import annotations
import numpy as np

class ExtendedKalmanFilterCWHPinhole:
    """Extended Kalman filter for 3-D CWH relative orbital motion using a Pinhole Camera.

    State vector  : x = [x, y, z, vx, vy, vz]^T
    Measurement   : z = [u, v, rho]^T  (focal plane offsets and range)
    """

    def __init__(
        self,
        mean_motion: float,
        focal_length: float,
        camera_offset_x: float = 0.0,
        dt: float = 60.0,
        process_noise_std: float = 1e-7,
        uv_noise_std: float = 7.5e-7,  # Noise in physical focal plane units (e.g. meters)
        range_noise_std: float = 1.0,
    ) -> None:
        self.n = mean_motion
        self.dt = dt
        self.f = focal_length
        self.D = camera_offset_x

        # 1. Construct the CWH State Transition Matrix (F)
        nt = self.n * self.dt
        s = np.sin(nt)
        c = np.cos(nt)
        n = self.n

        self.F = np.array([
            [4 - 3*c,       0.0,        0.0, s/n,          (2/n)*(1 - c), 0.0],
            [6*(s - nt),    1.0,        0.0, -(2/n)*(1-c), (4*s - 3*nt)/n, 0.0],
            [0.0,           0.0,        c,   0.0,          0.0,           s/n],
            [3*n*s,         0.0,        0.0, c,            2*s,           0.0],
            [-6*n*(1 - c),  0.0,        0.0, -2*s,         4*c - 3,       0.0],
            [0.0,           0.0,        -n*s,0.0,          0.0,           c]
        ])

        # 2. Process Noise Covariance (Piecewise constant acceleration model)
        q = process_noise_std ** 2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        
        # Applying a standard block diagonal Q matrix for the 3 Cartesian axes
        q_block = q * np.array([
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2]
        ])
        self.Q = np.zeros((6, 6))
        self.Q[0:4:3, 0:4:3] = q_block  # X and Vx
        self.Q[1:5:3, 1:5:3] = q_block  # Y and Vy
        self.Q[2:6:3, 2:6:3] = q_block  # Z and Vz

        # 3. Measurement Noise Covariance
        self.R = np.diag([uv_noise_std**2, uv_noise_std**2, range_noise_std**2])

    def _measurement_model(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the predicted measurement and the Jacobian H."""
        px, py, pz = x[0], x[1], x[2]
        
        # Depth is Y (in-track). Handle negative Y if the target is trailing.
        sign_y = 1.0 if py >= 0 else -1.0
        depth = abs(py)
        
        # Avoid division by zero
        denom = self.D + depth
        if abs(denom) < 1e-8:
            denom = 1e-8 

        # Non-linear measurement prediction h(x)
        u_pred = self.f * px / denom  # X maps to u (horizontal)
        v_pred = self.f * pz / denom  # Z maps to v (vertical)
        rho_pred = depth              # Y maps to range
        
        z_pred = np.array([u_pred, v_pred, rho_pred])

        # Jacobian H(x)
        H = np.zeros((3, 6))
        
        # d(u) / d(State)
        H[0, 0] = self.f / denom                          # du/dx
        H[0, 1] = -self.f * px / (denom**2) * sign_y      # du/dy
        
        # d(v) / d(State)
        H[1, 1] = -self.f * pz / (denom**2) * sign_y      # dv/dy
        H[1, 2] = self.f / denom                          # dv/dz
        
        # d(rho) / d(State)
        H[2, 1] = 1.0 * sign_y                            # drho/dy

        return z_pred, H

    def initialize(
        self,
        measurement: np.ndarray,
        velocity: np.ndarray | None = None,
        P0: np.ndarray | None = None,
        is_trailing: bool = True # Assumes target is trailing (negative Y)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize state from a [u, v, rho] measurement."""
        u, v, rho = measurement
        
        # Inverse mapping: from focal plane (u,v) + range back to 3D space
        denom = self.D + rho
        
        # Correctly map axes: Y is depth, X is horizontal, Z is vertical
        px = u * denom / self.f
        py = -rho if is_trailing else rho 
        pz = v * denom / self.f
        
        vel = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float)
        x = np.concatenate([[px, py, pz], vel])

        P0 = np.diag([1000, 1000, 1000, 10, 10, 10]) if P0 is None else np.asarray(P0, dtype=float)
            
        return x, np.asarray(P0, dtype=float)

    def predict(
        self, x: np.ndarray, P: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward using CWH dynamics."""
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_pred, P_pred

    def update(
        self,
        x_pred: np.ndarray,
        P_pred: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, dict]:
        """Incorporate a new [u, v, rho] measurement and return diagnostics."""
        z = np.asarray(z, dtype=float)
        
        z_pred, H = self._measurement_model(x_pred)
        
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        innovation = z - z_pred
        
        x_upd = x_pred + K @ innovation
        I = np.eye(len(x_pred))
        P_upd = (I - K @ H) @ P_pred

        n_dim = len(z)
        log_likelihood = -0.5 * (
            float(innovation @ np.linalg.solve(S, innovation))
            + np.log(np.linalg.det(S))
            + n_dim * np.log(2 * np.pi)
        )
        
        # --- NEW: Bundle the internal metrics for diagnostics ---
        kf_diagnostics = {
            "residual": innovation.copy(),
            "kalman_gain": K.copy(),
            "innovation_cov": S.copy()
        }
        
        return x_upd, P_upd, log_likelihood, kf_diagnostics

    def innovation_covariance(self, x_pred: np.ndarray, P_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Helper to compute S and H dynamically based on the current state."""
        _, H = self._measurement_model(x_pred)
        S = H @ P_pred @ H.T + self.R
        return S, H

    def mahalanobis_distance(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute the Mahalanobis distance using the non-linear measurement model."""
        # 1. Force everything to flat 1D arrays to prevent accidental 3x3 matrix generation
        z = np.asarray(z, dtype=float).flatten()
        z_pred, _ = self._measurement_model(x_pred)
        z_pred = np.asarray(z_pred, dtype=float).flatten()
        
        S, _ = self.innovation_covariance(x_pred, P_pred)
        innovation = z - z_pred
        
        # --- DEBUGGING BLOCK ---
        print(f"\n[DEBUG] Innovation (z - z_pred): {innovation}")
        print(f"[DEBUG] S Matrix Diagonal: {np.diag(S)}")
        # -----------------------
        
        try:
            # Solve S * x = innovation
            inv_S_innov = np.linalg.solve(S, innovation)
            md_sq = innovation @ inv_S_innov
            
            # Catch negative zeros or floating point anomalies 
            if md_sq < 0:
                return 0.0
                
            return float(np.sqrt(md_sq))
            
        except np.linalg.LinAlgError:
            # If S is singular or broken, reject the measurement entirely
            # print("[!] LinAlgError: S matrix is singular!")
            return 9999.0