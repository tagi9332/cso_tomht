def calculate_optical_properties(wavelength, f_len, D, pixel_pitch):
    """
    Calculates the Airy disk radius and PSF sigma in pixel units.
    """
    # Diffraction limit logic
    r_airy_m = 1.22 * wavelength * f_len / D
    r_airy_pix = r_airy_m / pixel_pitch
    
    # Gaussian approximation: sigma is roughly r_airy / 3
    sigma_psf_pix = r_airy_pix / 3.0
    
    return sigma_psf_pix, r_airy_pix