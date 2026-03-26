import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Your actual imports
from src.simulate_fits_data import simulate_fits_data
from src.process_fits_files import process_fits_directory
from src.process_centroids_to_tracks import process_centroids_to_tracks

## ==========================================
##  1. SIMULATION PIPELINE TESTS
## ==========================================

def test_simulate_fits_data(mocker):
    """Test that the simulation pipeline calls its sub-functions correctly."""
    mocker.patch("os.makedirs")
    mocker.patch("pandas.read_csv", return_value=pd.DataFrame())
    
    # Patch the functions inside the file where simulate_fits_data lives
    mock_run = mocker.patch(
        "src.simulate_fits_data.run_simulation", 
        return_value=(["fake_frame_1", "fake_frame_2"], pd.DataFrame())
    )
    
    mocker.patch("src.simulate_fits_data.export_frames")
    mocker.patch("src.simulate_fits_data.plot_summary_frame")
    mocker.patch("src.simulate_fits_data.create_simulation_gif")

    frames, truth_df = simulate_fits_data("dummy_trajectory.csv", verbose=False)

    mock_run.assert_called_once()
    assert len(frames) == 2


## ==========================================
##  2. FITS PROCESSING PIPELINE TESTS
## ==========================================

def test_process_fits_empty_dir(mocker):
    """Test that it safely returns if no FITS files are found."""
    mocker.patch("os.makedirs")
    mocker.patch("glob.glob", return_value=[])

    process_fits_directory("input", "out_csv", "out_plot", verbose=False)

def test_process_fits_success(mocker):
    """Test a full run of the image processing pipeline on mock FITS data."""
    mocker.patch("os.makedirs")
    mocker.patch("glob.glob", return_value=["frame1.fits"])
    mocker.patch("astropy.io.fits.getdata", return_value=np.zeros((10, 10)))
    
    # Patch the functions inside the file where process_fits_directory lives
    mocker.patch("src.process_fits_files.create_psf_kernel", return_value=np.ones((3,3)))
    mocker.patch("src.process_fits_files.levesque_process", return_value=(np.zeros((10,10)), None))
    mocker.patch("src.process_fits_files.detect_sources", return_value=[{'Centroid_X': 5, 'Centroid_Y': 5}])
    mocker.patch("src.process_fits_files.plot_detections", return_value="dummy.png")
    
    mocker.patch("pandas.DataFrame.to_csv")
    
    mock_image = mocker.patch("PIL.Image.open")
    mock_image.return_value.save = MagicMock()

    process_fits_directory("input", "out_csv", "out_plot", skip_bg_sub=False, verbose=False)


## ==========================================
##  3. TOMHT ORCHESTRATOR TESTS
## ==========================================

def test_process_centroids_missing_file(mocker):
    """Ensure it crashes gracefully if the CSV is missing."""
    mocker.patch("os.path.exists", return_value=False)
    
    with pytest.raises(FileNotFoundError):
        process_centroids_to_tracks("missing_data.csv")

def test_process_centroids_success(mocker):
    """Test that data is grouped and passed to the tracker correctly."""
    mocker.patch("os.path.exists", return_value=True)
    
    fake_csv = pd.DataFrame({
        'Frame_Idx': [1, 1], 
        'Centroid_X': [10.0, 20.0], 
        'Centroid_Y': [15.0, 25.0]
    })
    mocker.patch("pandas.read_csv", return_value=fake_csv)
    
    # Patch TrackerConfig and TOMHTTracker inside process_centroids_to_tracks
    mock_config = mocker.patch("src.process_centroids_to_tracks.TrackerConfig.from_jsonx").return_value
    mock_config.n_scan_window = 2

    mock_track = MagicMock()
    mock_track.track_id = 99
    mock_track.best_hypothesis.history_states = [[10.0, 15.0, 1.0, 1.0], [11.0, 16.0, 1.0, 1.0]]
    
    mock_tracker = mocker.patch("src.process_centroids_to_tracks.TOMHTTracker").return_value
    mock_tracker.step.return_value = [mock_track]

    result_df = process_centroids_to_tracks("fake_data.csv")

    assert not result_df.empty
    assert result_df.iloc[0]['id'] == 99
    assert result_df.iloc[0]['x'] == 10.0