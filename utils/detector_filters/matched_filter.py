import numpy as np
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any

def create_psf_kernel(sigma_psf: float) -> np.ndarray:
    """
    Generates a normalized 2D Gaussian kernel for matched filtering.
    """
    kernel_rad = int(3 * sigma_psf)
    y_k, x_k = np.mgrid[-kernel_rad:kernel_rad+1, -kernel_rad:kernel_rad+1]
    psf_kernel = np.exp(-(x_k**2 + y_k**2) / (2 * sigma_psf**2))
    psf_kernel /= np.sum(psf_kernel)
    
    return psf_kernel

def detect_sources(
    img_clean: np.ndarray, 
    psf_kernel: np.ndarray, 
    sigma_psf: float, 
    threshold_factor: float, 
    frame_idx: int, 
    filename: str
) -> List[Dict[str, Any]]:
    """
    Applies a matched filter to a cleaned image, calculates robust noise statistics, 
    and uses DBSCAN to cluster and centroid detections.
    """
    # Matched Filter
    score_map = convolve2d(img_clean, psf_kernel, mode='same')
    
    # Robust Thresholding (MAD)
    median_score = np.median(score_map)
    robust_sigma = 1.4826 * np.median(np.abs(score_map - median_score))
    if robust_sigma == 0: robust_sigma = 1e-6 
    
    threshold = threshold_factor * robust_sigma
    bright_indices = np.argwhere(score_map > threshold)
    
    frame_detections = []
    
    # Clustering & Centroiding
    if len(bright_indices) > 0:
        clustering = DBSCAN(eps=3.0, min_samples=2).fit(bright_indices)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1} 
        
        for label in unique_labels:
            cluster_mask = (labels == label)
            points = bright_indices[cluster_mask]
            ys, xs = points[:, 0], points[:, 1]
            
            cluster_scores = score_map[ys, xs]
            total_score = np.sum(cluster_scores)
            
            if total_score > 0:
                # Intensity-weighted average
                y_c = np.sum(ys * cluster_scores) / total_score
                x_c = np.sum(xs * cluster_scores) / total_score
                
                peak_val = np.max(cluster_scores)
                snr = peak_val / robust_sigma
                
                # Variance calculation (Cramer-Rao Lower Bound approx)
                pos_variance = (sigma_psf / snr)**2 if snr > 0 else 0.0
                
                frame_detections.append({
                    'Frame_Idx': frame_idx,
                    'Filename': filename,
                    'Centroid_X': x_c,
                    'Centroid_Y': y_c,
                    'Cov_XX': pos_variance,
                    'Cov_YY': pos_variance,
                    'Cov_XY': 0.0, 
                    'SNR': snr,
                    'Peak_Value': peak_val,
                    'Cluster_Size': len(points)
                })
                
    return frame_detections