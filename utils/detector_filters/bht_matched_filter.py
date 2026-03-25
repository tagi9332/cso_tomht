import numpy as np
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN
from scipy.stats import norm

class MatchedFilterDetector:
    def __init__(self, sigma_psf, kernel_ratio=4.0):
        self.sigma_psf = sigma_psf
        self.kernel = self._create_kernel(sigma_psf, kernel_ratio)
        self.kernel_norm_factor = np.sqrt(np.sum(self.kernel**2))
        self.amp_scale_factor = np.max(self.kernel) / np.sum(self.kernel**2)

    def _create_kernel(self, sigma, ratio):
        r = int(ratio * sigma)
        y, x = np.mgrid[-r:r+1, -r:r+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def detect_bht(self, subtracted_image, pixel_noise, p_fa):
        """
        Detects objects using Binary Hypothesis Testing based on a target P_FA.
        Returns list of dicts containing centroid, s_hat, sigma_s, and sigma_c.
        """
        # --- DETECTION PHASE (Matched Filter) ---
        score_map = convolve2d(subtracted_image, self.kernel, mode='same')
        sigma_mf = pixel_noise * self.kernel_norm_factor
        
        threshold_multiplier = norm.isf(p_fa)
        threshold = threshold_multiplier * sigma_mf * 10**-2
        
        bright_indices = np.argwhere(score_map > threshold)
        if len(bright_indices) == 0:
            return []

        clustering = DBSCAN(eps=3.0, min_samples=1).fit(bright_indices)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        detections = []
        
        # Define the tracking window radius (e.g., 3 * sigma_psf to capture ~99% of signal)
        win_r = int(np.ceil(3 * self.sigma_psf))
        
        # --- ESTIMATION PHASE (Tracking Window) ---
        for label in unique_labels:
            # 1. Get rough center from the DBSCAN cluster to place our window
            mask = (labels == label)
            cluster_coords = bright_indices[mask]
            rough_y = int(np.mean(cluster_coords[:, 0]))
            rough_x = int(np.mean(cluster_coords[:, 1]))
            
            # 2. Define tracking window boundaries, ensuring we don't go off the image edge
            y_min = max(0, rough_y - win_r)
            y_max = min(subtracted_image.shape[0], rough_y + win_r + 1)
            x_min = max(0, rough_x - win_r)
            x_max = min(subtracted_image.shape[1], rough_x + win_r + 1)
            
            # Extract the window from the SUBTRACTED image (not the matched filter score_map)
            window = subtracted_image[y_min:y_max, x_min:x_max]
            y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
            
            # --- CALCULATE REQUIRED METRICS ---
            
            # Photometric Estimate (s_hat): Total signal in tracking window
            s_hat = np.sum(window)
            
            if s_hat > 0:
                # Centroid Estimate (x_c, y_c): Intensity-weighted average
                y_c = np.sum(y_grid * window) / s_hat
                x_c = np.sum(x_grid * window) / s_hat
                
                # Photometric Uncertainty (sigma_s)
                # Assuming background-limited noise: variance adds up per pixel in the window
                N_w = window.size
                sigma_s = pixel_noise * np.sqrt(N_w)
                
                # Centroid Uncertainty (sigma_c)
                # Cramer-Rao Lower Bound approximation: sigma_c = sigma_psf / SNR
                snr = s_hat / sigma_s if sigma_s > 0 else 1e-9
                sigma_c = self.sigma_psf / snr
                
                detections.append({
                    'x': x_c, 
                    'y': y_c, 
                    's_hat': s_hat, 
                    'sigma_s': sigma_s, 
                    'sigma_c': sigma_c
                })
                
        return detections