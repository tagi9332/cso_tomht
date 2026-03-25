import numpy as np
import cv2
''' Levesque background subtraction implementation '''

def levesque_process(image, sigma_psf=2.0, k=2.5, iterations=5):
    img_float = image.astype(np.float32)
    rows, cols = img_float.shape
    min_dim = min(rows, cols)
    
    spot_diam = 6 * sigma_psf
    raw_large = int(20 * spot_diam)
    raw_small = int(10 * spot_diam)
    
    max_window = int(min_dim / 5)
    size_large = min(raw_large, max_window)
    size_small = min(raw_small, max_window)
    
    if size_large % 2 == 0: size_large += 1
    if size_small % 2 == 0: size_small += 1
    
    background = np.full_like(img_float, np.median(img_float))
    
    for _ in range(iterations):
        residual = img_float - background
        sigma_curr = np.std(residual)
        if sigma_curr == 0: sigma_curr = 1e-6 
        
        mask = (img_float < (background + k * sigma_curr)).astype(np.float32)
        masked_img = img_float * mask
        
        smooth_numer = cv2.boxFilter(masked_img, -1, (size_large, size_large), normalize=False)
        smooth_denom = cv2.boxFilter(mask, -1, (size_large, size_large), normalize=False)
        smooth_denom[smooth_denom == 0] = 1.0
        
        bg_update = smooth_numer / smooth_denom
        background = cv2.boxFilter(bg_update, -1, (size_small, size_small), normalize=True)
        
    return img_float - background, background