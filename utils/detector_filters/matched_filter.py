import numpy as np
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN

# Class definition
class MatchedFilterDetector:
    def __init__(self, sigma_psf, kernel_ratio=4.0):
        """
        Args:
            sigma_psf (float): The PSF sigma in pixels.
            kernel_ratio (float): How large the kernel should be relative to sigma (radius).
        """
        self.sigma_psf = sigma_psf
        self.kernel = self._create_kernel(sigma_psf, kernel_ratio)
        self.kernel_norm_factor = np.sqrt(np.sum(self.kernel**2))

    def _create_kernel(self, sigma, ratio):
        r = int(ratio * sigma)
        y, x = np.mgrid[-r:r+1, -r:r+1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)

    def detect(self, subtracted_image, pixel_noise, threshold_factor=4.0):
        """
        Runs matched filtering on a pre-subtracted image.
        Returns list of {'x': float, 'y': float}
        """
        # 1. Convolution (Matched Filtering) directly on the subtracted image
        score_map = convolve2d(subtracted_image, self.kernel, mode='same')
        
        # 2. Calculate Threshold
        score_noise = pixel_noise * self.kernel_norm_factor
        threshold = threshold_factor * score_noise
        
        # 3. Find Bright Pixels
        bright_indices = np.argwhere(score_map > threshold)
        
        if len(bright_indices) == 0:
            return []

        # 4. Cluster (DBSCAN)
        clustering = DBSCAN(eps=3.0, min_samples=1).fit(bright_indices)
        labels = clustering.labels_
        unique_labels = set(labels)
        if -1 in unique_labels: unique_labels.remove(-1)

        detections = []
        for label in unique_labels:
            mask = (labels == label)
            cluster_coords = bright_indices[mask]
            ys = cluster_coords[:, 0]
            xs = cluster_coords[:, 1]
            
            # Weighted Centroid using Score Map values
            weights = score_map[ys, xs]
            total_w = np.sum(weights)
            
            if total_w > 0:
                y_c = np.sum(ys * weights) / total_w
                x_c = np.sum(xs * weights) / total_w
                detections.append({'x': x_c, 'y': y_c})
                
        return detections

