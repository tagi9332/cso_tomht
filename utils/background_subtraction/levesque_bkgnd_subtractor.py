import numpy as np
import cv2
from scipy.optimize import curve_fit

# --- CLASS DEFINITION ---

class LevesqueBackgroundSubtractor:
    def __init__(self, k=3.0, iterations=5):
        """
        Initializes the Levesque Background Subtractor.
        
        Args:
            k (float): Sigma multiplier for the thresholding mask (default 3.0).
            iterations (int): Number of smoothing iterations (default 5).
        """
        self.k = k
        self.iterations = iterations

    @staticmethod
    def _gaussian_2d(xy, amplitude, x0, y0, sigma, offset):
        """Internal helper for curve_fit."""
        x, y = xy
        g = offset + amplitude * np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))
        return g.ravel()

    def estimate_psf_sigma(self, image):
        """
        Estimates the PSF Sigma by fitting a 2D Gaussian to the brightest spot.
        Returns: (sigma_est, (center_x, center_y))
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        # Find brightest spot
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image_gray)
        cx, cy = max_loc

        # Crop around the spot
        r = 10
        h, w = image_gray.shape
        y_min, y_max = max(0, cy-r), min(h, cy+r)
        x_min, x_max = max(0, cx-r), min(w, cx+r)
        crop = image_gray[y_min:y_max, x_min:x_max]
        
        if crop.size == 0:
            return 2.0, (cx, cy)

        # Grid for fitting
        ch, cw = crop.shape
        x_grid, y_grid = np.meshgrid(np.arange(cw), np.arange(ch))
        
        initial_guess = (crop.max(), cw/2, ch/2, 2.0, crop.min())
        
        try:
            popt, _ = curve_fit(self._gaussian_2d, (x_grid, y_grid), crop.ravel(), p0=initial_guess)
            sigma_est = abs(popt[3])
            # Sanity check: if fit is wild, clamp it
            if sigma_est < 0.5 or sigma_est > 20:
                sigma_est = 2.0
            return sigma_est, (cx, cy)
        except Exception as e:
            # Fallback if fit fails
            return 2.0, (cx, cy)

    def process(self, image, manual_sigma=None):
        """
        Performs the background subtraction.
        
        Args:
            image (np.ndarray): Input image.
            manual_sigma (float): Optional. Force a specific PSF sigma.
            
        Returns:
            dict: {
                'background': np.ndarray,
                'subtracted': np.ndarray,
                'sigma_b': float (std of result),
                'sigma_psf': float (estimated or manual),
                'psf_loc': tuple (x, y)
            }
        """
        # Prepare Data
        img_float = image.astype(np.float32)
        rows, cols = img_float.shape
        min_dim = min(rows, cols)

        # 1. Determine PSF Sigma
        if manual_sigma is not None:
            sigma_psf = manual_sigma
            # We still run this just to find the location for plotting later
            _, psf_loc = self.estimate_psf_sigma(image)
        else:
            sigma_psf, psf_loc = self.estimate_psf_sigma(image)

        # 2. Determine Window Sizes based on Levesque logic
        spot_diam = 6 * sigma_psf
        
        raw_large = int(20 * spot_diam)
        raw_small = int(10 * spot_diam)
        
        # Clamp: Window never larger than 1/4th of image
        max_window = int(min_dim / 4)
        
        size_large = min(raw_large, max_window)
        size_small = min(raw_small, max_window)
        
        # Ensure odd
        if size_large % 2 == 0: size_large += 1
        if size_small % 2 == 0: size_small += 1
        
        # Prevent window from being too small (must be at least 3)
        size_large = max(3, size_large)
        size_small = max(3, size_small)

        # 3. Iterative Background Estimation
        background = np.full_like(img_float, np.median(img_float))
        
        for i in range(self.iterations):
            # Residual statistics
            residual = img_float - background
            sigma_curr = np.std(residual)
            
            # Mask sources (values > background + k*sigma)
            # We use < to create a mask of "background pixels" (1=bg, 0=source)
            mask = (img_float < (background + self.k * sigma_curr)).astype(np.float32)
            
            # Update Background using masked average
            masked_img = img_float * mask
            
            # Large Window Smoothing
            # boxFilter is faster than manual convolution
            smooth_numer = cv2.boxFilter(masked_img, -1, (size_large, size_large), normalize=False)
            smooth_denom = cv2.boxFilter(mask, -1, (size_large, size_large), normalize=False)
            
            # Avoid division by zero
            smooth_denom[smooth_denom == 0] = 1.0
            bg_update = smooth_numer / smooth_denom
            
            # Small Window Refinement (Smoothing the background estimate itself)
            bg_update = cv2.boxFilter(bg_update, -1, (size_small, size_small), normalize=True)
            
            background = bg_update

        # 4. Final Subtraction
        final_subtracted = img_float - background
        final_sigma_b = np.std(final_subtracted)
        
        return {
            'background': background,
            'subtracted': final_subtracted,
            'sigma_b': final_sigma_b,
            'sigma_psf': sigma_psf,
            'psf_loc': psf_loc,
            'window_large': size_large,
            'window_small': size_small
        }