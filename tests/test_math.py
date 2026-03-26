import pytest
import numpy as np

# Adjust these imports based on your actual file structure
from utils.kalman_filter import KalmanFilter2D
from utils.kdtree_association import KDTreeAssociation

## ==========================================
##  KALMAN FILTER TESTS
## ==========================================

@pytest.fixture
def kf():
    """Provides a basic 2D Kalman Filter."""
    return KalmanFilter2D(dt=1.0, process_noise_std=0.1, measurement_noise_std=1.0)

def test_kf_predict_constant_velocity(kf):
    """Test that the KF correctly projects the state forward based on velocity."""
    # State: [x=10, y=10, vx=2, vy=-1]
    initial_state = np.array([10.0, 10.0, 2.0, -1.0])
    initial_covariance = np.eye(4)
    
    x_pred, P_pred = kf.predict(initial_state, initial_covariance)
    
    # After dt=1.0, new position should be [12, 9], velocities unchanged
    assert np.allclose(x_pred, [12.0, 9.0, 2.0, -1.0])
    
    # Covariance should grow due to process noise
    assert np.trace(P_pred) > np.trace(initial_covariance)

def test_kf_mahalanobis_distance(kf):
    """Test that the Mahalanobis distance scales correctly with covariance."""
    state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Tight covariance (high certainty)
    tight_cov = np.eye(4) * 0.1
    # Loose covariance (low certainty)
    loose_cov = np.eye(4) * 10.0
    
    measurement = np.array([2.0, 2.0])
    
    dist_tight = kf.mahalanobis_distance(state, tight_cov, measurement)
    dist_loose = kf.mahalanobis_distance(state, loose_cov, measurement)
    
    # The same physical distance should result in a HIGHER Mahalanobis distance 
    # if the tracker is highly certain of its position (tight covariance).
    assert dist_tight > dist_loose


## ==========================================
##  KD-TREE ASSOCIATION TESTS
## ==========================================

@pytest.fixture
def kdtree():
    """Provides a KD-Tree with a 10-pixel gate."""
    return KDTreeAssociation(gate_distance=10.0, merge_clusters=False)

def test_kdtree_clustering_within_gate(kdtree):
    """Test that points close to each other are clustered together."""
    tracks = np.array([[0, 0], [100, 100]])
    
    # Meas 1 is at [1,1] (close to track 0)
    # Meas 2 is at [101, 101] (close to track 1)
    measurements = np.array([[1, 1], [101, 101]])
    
    clusters = kdtree.cluster(tracks, measurements)
    
    # Should form 2 distinct clusters
    assert len(clusters) == 2
    
    # Track 0 should only be associated with Measurement 0
    assert 0 in clusters[0]["track_indices"]
    assert 0 in clusters[0]["meas_indices"]
    assert 1 not in clusters[0]["track_indices"]

def test_kdtree_gate_rejection(kdtree):
    """Test that measurements outside the gate distance are NOT associated."""
    tracks = np.array([[0, 0]])
    
    # Measurement is 50 pixels away (gate is 10)
    measurements = np.array([[50, 50]])
    
    clusters = kdtree.cluster(tracks, measurements)
    
    # As long as the measurement is outside the gate, it should not be associated with the track:
    # 1. Two separate clusters (one for the track, one for the unassociated measurement)
    # 2. An empty track list for the measurement cluster
    
    for cluster in clusters:
        # Should never see track 0 and measurement 0 in the same cluster
        if 0 in cluster["track_indices"]:
            assert 0 not in cluster["meas_indices"]