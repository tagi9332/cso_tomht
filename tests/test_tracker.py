import pytest
import numpy as np
from unittest.mock import MagicMock

# Import your classes (adjust paths as necessary)
from src.tomht import TOMHTTracker
from src.track import Track, Hypothesis

## --- FIXTURES ---

@pytest.fixture
def mock_config():
    """Provides a basic tracker configuration using MagicMock."""
    config = MagicMock()
    config.dt = 1.0
    config.kf_process_noise = 0.1
    config.kf_meas_noise = 1.0
    config.gate_distance = 10.0
    config.merge_clusters = True
    config.max_hypotheses = 5
    config.miss_log_likelihood = -5.0
    config.n_scan_window = 2
    config.mahalanobis_thresh = 9.48 
    config.max_misses = 3
    config.min_age_to_check = 5
    config.min_distance_px = 5.0
    return config

@pytest.fixture
def sample_track(mock_config):
    """Initializes a single track for isolated testing."""
    return Track(
        track_id=0,
        initial_state=np.array([10, 10, 1, 1]),
        initial_covariance=np.eye(4),
        max_hypotheses=mock_config.max_hypotheses,
        n_scan_window=mock_config.n_scan_window
    )

## --- TRACK DATA STRUCTURE TESTS ---

def test_track_initialization(sample_track):
    """Ensure a track starts with exactly one root hypothesis."""
    assert len(sample_track.hypotheses) == 1
    assert sample_track.track_id == 0
    assert np.array_equal(sample_track.start_pos, [10, 10])

def test_hypothesis_expansion(sample_track):
    """Test that a track expands into Detection and Null branches."""
    # We simulate 1 parent hypothesis expanding into 2 new hypotheses (1 null, 1 detection)
    preds = [(np.array([11, 11, 1, 1]), np.eye(4))]
    updates = [[
        (None, np.array([11, 11, 1, 1]), np.eye(4), -2.0), # Null hypothesis
        (0, np.array([11.5, 11.5, 1, 1]), np.eye(4), -0.5) # Detection hypothesis
    ]]
    
    sample_track.expand_hypotheses(preds, updates)
    
    assert len(sample_track.hypotheses) == 2
    # Ensure IDs are unique
    assert sample_track.hypotheses[0].hyp_id != sample_track.hypotheses[1].hyp_id
    # Check that lineage history grew
    assert len(sample_track.hypotheses[0].history_ids) == 2

def test_n_scan_pruning(sample_track):
    """Crucial: Ensure competing realities collapse to the best ancestor."""
    sample_track.n_scan_window = 2
    sample_track.age = 3
    
    # Manually create 2 hypotheses with DIFFERENT ancestors from 2 steps ago to force a collapse.
    # Hyp 1 (Winner): History [Root, Node1, Node2] -> Ancestor 2 steps ago is ID '1'
    # Hyp 2 (Loser): History [Root, Node99, Node100] -> Ancestor 2 steps ago is ID '99'
    
    h1 = Hypothesis(hyp_id=10, state=np.zeros(4), covariance=np.eye(4), 
                    log_score=10.0, history_ids=[0, 1, 2])
    h2 = Hypothesis(hyp_id=11, state=np.zeros(4), covariance=np.eye(4), 
                    log_score=5.0, history_ids=[0, 99, 100])
    
    sample_track.hypotheses = [h1, h2]
    
    # Run the N-Scan pruner
    sample_track.apply_n_scan_pruning()
    
    # Only h1 should survive because it has the highest score, and h2 does not share its N-scan ancestor.
    assert len(sample_track.hypotheses) == 1
    assert sample_track.hypotheses[0].hyp_id == 10


## --- TRACKER LOGIC TESTS ---

def test_tracker_initiation(mock_config, mocker):
    """Test that an unassociated measurement correctly spawns a new track."""
    # Mock dependencies inside __init__
    mocker.patch('utils.kalman_filter.KalmanFilter2D')
    mocker.patch('utils.kdtree_association.KDTreeAssociation')
    
    tracker = TOMHTTracker(mock_config)
    
    # Step with one mock measurement
    measurements = np.array([[50, 50]])
    tracker.step(measurements)
    
    assert len(tracker.active_tracks) == 1
    # Check that the track was initiated at the measurement position
    assert tracker.active_tracks[0].start_pos[0] == 50

def test_tracker_miss_pruning(mock_config, mocker):
    """Test that tracks are aggressively deleted after exceeding max_misses."""
    mocker.patch('utils.kalman_filter.KalmanFilter2D')
    mocker.patch('utils.kdtree_association.KDTreeAssociation')
    
    tracker = TOMHTTracker(mock_config)
    tracker._initiate_track(np.array([10, 10]))
    
    track = tracker.active_tracks[0]
    track.consecutive_misses = mock_config.max_misses - 1
    
    # Provide no measurements, triggering a consecutive miss
    tracker.step(np.array([]))
    
    # Track should now be pruned out of active_tracks
    assert len(tracker.active_tracks) == 0
