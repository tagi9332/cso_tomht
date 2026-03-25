*******************************************************************************
* CLOSELY-SPACED OBJECT (CSO) TOMHT PIPELINE                           *
* Simulation | Detection | Multi-Hypothesis Tracking               *
*******************************************************************************

===============================================================================
1. OVERVIEW
===============================================================================
An end-to-end modular pipeline designed to simulate astronomical image data,
extract dim source centroids, and maintain target identities using a
Track-Oriented Multiple Hypothesis Tracker (TOMHT).

The pipeline consists of three primary stages:
  [SIMULATION] -> [DETECTION] -> [TRACKING]

===============================================================================
2. INSTALLATION & REQUIREMENTS
===============================================================================
Tested on Python 3.12+. Required packages:
  > pip install -r requirements.txt

===============================================================================
3. USAGE
===============================================================================
The entire workflow is automated via main.py.

  1. Set your input file in main.py:
     TARGET_TRAJECTORY_CSV = "data/cso_data/traj_file.csv"

  2. Execute the script:
     python main.py

===============================================================================
4. PIPELINE STAGES
===============================================================================

[STAGE 1: SIMULATION]
  Converts trajectories into .fits frames.
  - Applies Airy disk PSF assumption.
  - Injects Poisson photon noise and Gaussian read noise.

[STAGE 2: DETECTION]
  Extracts (x, y) centroids from noisy imagery.
  - Levesque Background Subtraction: Removes non-uniform offsets.
  - Matched Filter: Convolves image with a Gaussian kernel to boost SNR.
  - DBSCAN Clustering: Groups local maxima into discrete detections.

[STAGE 3: TRACKING (TOMHT)]
  Associates detections across time while resolving identity ambiguities.
  - Gating: Uses KD-Trees for O(log n) nearest-neighbor association.
  - N-Scan Pruning: Efficiently manages the hypothesis tree window.
  - Kalman Filtering: Estimates position and velocity states.

===============================================================================
5. KEY OUTPUTS (Found in /results)
===============================================================================
  FILE                          DESCRIPTION
  ---------------------------------------------------------------------------
  master_detections.csv         Full list of all centroids found in frames.
  detections_animation.gif      Visual confirmation of raw detections.
  tomht_animation.gif           Final tracked paths with IDs & velocities.
  tomht_static.png              Summary of the longest tracks found.

*******************************************************************************
* END OF DOCUMENT                                    *
*******************************************************************************