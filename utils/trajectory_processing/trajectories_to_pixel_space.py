import numpy as np
import pandas as pd

def trajectories_to_pixel_space(trajectory_files: dict, config: dict, earth_radius_km: float = 6378.137) -> pd.DataFrame:
    """
    Ingests trajectory CSVs and converts RIC frame relative motion into 2D pixel coordinates.
    
    Args:
        trajectory_files (dict): Dictionary mapping object IDs to their CSV file paths.
                                 e.g., {"Chief": "chief.csv", "Deputy_A": "dep_A.csv"}
        config (dict): Full configuration dictionary loaded from JSON, containing the 
                       'optical_sensor' block with:
                       - 'f_len': focal length in meters
                       - 'pixel_pitch': physical size of a pixel in meters
                       - 'img_size': width/height of the sensor in pixels
        earth_radius_km (float): Radius of the Earth to subtract for a surface observer.
        
    Returns:
        pd.DataFrame: Formatted dataframe with columns ['time', 'id', 'x', 'y']
    """
    
    # Extract the optical sensor configuration (with a fallback if the sub-dict is passed directly)
    sensor_config = config.get('optical_sensor', config)
    
    # Calculate the Instantaneous Field of View (IFoV) in radians per pixel
    # IFoV = pixel size / focal length
    ifov = sensor_config['pixel_pitch'] / sensor_config['f_len']
    
    # Center coordinate of the sensor
    center_x = sensor_config['img_size'] / 2.0
    center_y = sensor_config['img_size'] / 2.0
    
    master_records = []
    
    for obj_id, filepath in trajectory_files.items():
        # Read the trajectory data
        df = pd.read_csv(filepath)
        
        # Calculate the magnitude of the inertial position vector (Earth center to Chief/Deputy)
        # Assuming r_x_km, r_y_km, r_z_km are in the ECI frame
        r_mag_km = np.sqrt(df['r_x_km']**2 + df['r_y_km']**2 + df['r_z_km']**2)
        
        # Calculate distance from the observer (Earth surface) to the target
        distance_to_target_km = r_mag_km - earth_radius_km
        
        # Calculate the angular offset in radians using small angle approximation (theta ~ rho/D)
        # rho_y_km (In-track) maps to the horizontal axis of the sensor
        # rho_z_km (Cross-track) maps to the vertical axis of the sensor
        theta_y = df['rho_y_km'] / distance_to_target_km
        theta_z = df['rho_z_km'] / distance_to_target_km
        
        # Convert angular offset to pixel offset
        pixel_offset_x = theta_y / ifov
        pixel_offset_y = theta_z / ifov
        
        # Shift relative to the center of the camera sensor
        pixel_x = center_x + pixel_offset_x
        pixel_y = center_y + pixel_offset_y
        
        # Build the temporary dataframe for this object
        obj_df = pd.DataFrame({
            'time': df['time_s'],
            'id': obj_id,
            'x': pixel_x,
            'y': pixel_y
        })
        
        master_records.append(obj_df)
        
    # Concatenate all objects into a single master dataframe
    formatted_trajectory_df = pd.concat(master_records, ignore_index=True)
    
    # Sort by time so the simulation can step through chronologically
    formatted_trajectory_df.sort_values(by=['time', 'id'], inplace=True)
    formatted_trajectory_df.reset_index(drop=True, inplace=True)
    
    return formatted_trajectory_df