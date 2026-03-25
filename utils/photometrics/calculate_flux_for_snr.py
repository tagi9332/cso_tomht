import numpy as np

def calculate_flux_for_snr(target_snr, n_pix, bg, rn):
    """
    Solves the specific photometric SNR equation for signal (S).
    Equation: SNR = S / sqrt(m*rn^2 + S + m*bg)
    """
    # Variance from read noise and background over m pixels
    total_noise_variance = n_pix * (rn**2 + bg)
    
    # Quadratic Coefficients: a*S^2 + b*S + c = 0
    a = 1.0
    b_coeff = -(target_snr ** 2)
    c = -(target_snr ** 2) * total_noise_variance
    
    # Solve using quadratic formula: S = (-b + sqrt(b^2 - 4ac)) / 2a
    discriminant = b_coeff**2 - 4 * a * c
    S = (-b_coeff + np.sqrt(discriminant)) / (2 * a)
    
    return S