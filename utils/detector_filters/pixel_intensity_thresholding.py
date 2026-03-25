import numpy as np
from sklearn.cluster import DBSCAN

class pixel_threshold_detector:
    def __init__(self, threshold_sigma=3.0, min_cluster_size=3, dbscan_eps=2.0):
        """
        A detector that uses pixel intensity thresholding followed by DBSCAN clustering.
        
        Args:
            threshold_sigma (float): How many standard deviations above background to set threshold.
            min_cluster_size (int): Minimum number of pixels required to form a valid detection.
            dbscan_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        """
        self.threshold_sigma = threshold_sigma
        self.min_cluster_size = min_cluster_size
        self.dbscan_eps = dbscan_eps
        
        # Initialize DBSCAN once to save overhead if parameters don't change
        self.clustering_algo = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_cluster_size)

    def detect(self, frame, bg_mean, read_noise_std):
        """
        Process a single frame and return detected (x, y) centroids.
        
        Args:
            frame (np.ndarray): The 2D image array.
            bg_mean (float): The mean background count level.
            read_noise_std (float): The standard deviation of the read noise.
            
        Returns:
            list of dict: [{'x': float, 'y': float}, ...]
        """
        # 1. Calculate Threshold
        # Noise floor = Poisson Shot Noise from Sky + Gaussian Read Noise
        # sigma_total = sqrt(sigma_sky^2 + sigma_read^2) -> sqrt(bg_mean + rn^2)
        noise_sigma = np.sqrt(bg_mean + read_noise_std**2)
        threshold_value = bg_mean + (self.threshold_sigma * noise_sigma)
        
        # 2. Identify Bright Pixels
        # Returns indices as [[y1, x1], [y2, x2], ...]
        bright_pixel_indices = np.argwhere(frame > threshold_value)
        
        detections = []
        
        # If no pixels are bright enough, return empty list
        if len(bright_pixel_indices) == 0:
            return detections
            
        # 3. Cluster Pixels (DBSCAN)
        # We fit only on the coordinates of the bright pixels
        labels = self.clustering_algo.fit_predict(bright_pixel_indices)
        
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1) # -1 represents noise in DBSCAN
            
        # 4. Calculate Centroids
        for label in unique_labels:
            # Get mask for current cluster
            cluster_mask = (labels == label)
            cluster_coords = bright_pixel_indices[cluster_mask]
            
            # Extract coordinates
            ys = cluster_coords[:, 0]
            xs = cluster_coords[:, 1]
            
            # Extract intensity values for weighting
            # Weight = Pixel Value - Background
            pixel_values = frame[ys, xs]
            weights = pixel_values - bg_mean
            
            # Clip negative weights (just in case of noise fluctuations near threshold)
            weights = np.maximum(0, weights)
            total_weight = np.sum(weights)
            
            if total_weight > 0:
                y_w = np.sum(ys * weights) / total_weight
                x_w = np.sum(xs * weights) / total_weight
                
                detections.append({'x': x_w, 'y': y_w})
            else:
                # Fallback to geometric center if weights are zero
                detections.append({'x': np.mean(xs), 'y': np.mean(ys)})
                
        return detections