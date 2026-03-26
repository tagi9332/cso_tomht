# Closely-Spaced Object (CSO) TOMHT Pipeline
**Simulation | Detection | Multi-Hypothesis Tracking**

## Overview
The CSO TOMHT Pipeline is an end-to-end modular framework designed to simulate astronomical image data, extract dim source centroids, and maintain target identities using a Track-Oriented Multiple Hypothesis Tracker (TOMHT). 

## Pipeline Architecture
The workflow is fully automated and consists of three primary stages:

### 1. Simulation
Converts trajectories into synthetic `.fits` frames.
* Applies an Airy disk Point Spread Function (PSF) assumption.
* Injects realistic Poisson photon noise and Gaussian read noise.

### 2. Detection
Extracts (x, y) centroids from noisy imagery.
* **Levesque Background Subtraction:** Removes non-uniform background offsets.
* **Matched Filter:** Convolves the image with a Gaussian kernel to boost the Signal-to-Noise Ratio (SNR).
* **DBSCAN Clustering:** Groups local maxima into discrete, trackable detections.

### 3. Tracking (TOMHT)
Associates detections across time while natively resolving target identity ambiguities during crossovers.
* **Gating:** Utilizes KD-Trees for highly efficient nearest-neighbor association.
* **N-Scan Pruning:** Manages the hypothesis tree memory window to prevent combinatorial explosions.
* **Kalman Filtering:** Estimates and predicts position and velocity states for all tracks.

## Key Outputs
Upon completion, the pipeline generates the following artifacts in the `/results` directory:

* `master_detections.csv`: Full list of all centroids found across all frames.
* `detections_animation.gif`: Visual confirmation of raw pipeline detections.
* `tomht_animation.gif`: Final tracked paths overlaid with track IDs and velocities.
* `tomht_static.png`: Plot summarizing the longest continuous tracks found.

## Getting Started

1. Ensure you are running **Python 3.12+**. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt