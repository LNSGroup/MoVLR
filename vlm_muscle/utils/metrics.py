import numpy as np
from scipy.stats import linregress
from .utils import file_to_string

def calculate_success(args, task_type, qpos_file):
    # read in qpos data, convert to list of lists, each inner list is a timestep
    qpos_data = file_to_string(qpos_file)
    qpos_data = [list(map(float, row.split(','))) for row in qpos_data.split('\n')[:-1]]
    pelvis_series = [row[0:6] for row in qpos_data]
    
    if task_type == "locomotion":
        """success criterion for locomotion tasks: 
            - not falling (pelvis height > 0.35)
            - walking forward (pelvis orientation does not deviate further than 25 degrees of forward throughout)
            - walking with fast enough speed (distance traveled in desired direction / total simulation time)
        """
        forward = is_walking_forward(pelvis_series)
        fallen  = has_fallen(pelvis_series)
        deviate = has_deviated(pelvis_series, threshold=0.3)
        rotated = has_rotated(pelvis_series, threshold=0.5) # radians, about 30 degrees
        speed, fast_enough  = get_speed(pelvis_series, args)
        
        print(f"forward: {forward}, \nfallen: {fallen}, \ndeviate: {deviate}, \nrotated: {rotated}, \nspeed: {speed}, \nfast_enough: {fast_enough}")
        
        success = forward and (not fallen) and (not deviate) and (not rotated) and fast_enough
        
        return success
        
def is_walking_forward(pelvis_series, min_slope=0.001, min_rvalue=0.5):
    """
    Determines if the model is walking forward (positive x-direction) over time.

    Args:
        pelvis_series (list of list of float): Each element is a 6-element list of pelvis joint states.
        min_slope (float): Minimum required slope in pelvis_tx to consider forward walking.
        min_rvalue (float): Minimum correlation coefficient to indicate consistent forward movement.
    """
    if len(pelvis_series) < 3:
        return False  # Too little data

    pelvis_tx_values = [frame[2] for frame in pelvis_series]
    timesteps = np.arange(len(pelvis_tx_values))

    # Linear regression: pelvis_tx = slope * timestep + intercept
    slope, intercept, r_value, p_value, std_err = linregress(timesteps, pelvis_tx_values)

    return slope > min_slope and abs(r_value) > min_rvalue

def has_fallen(pelvis_series, threshold=-0.2, debug=False):
    """
    Determines if the model has fallen by checking if pelvis_ty drops below a threshold.

    Args:
        pelvis_series (list of list of float): Each element is a 6-element list of pelvis joint states.
        threshold (float): The pelvis_ty value below which the model is considered to have fallen.
        
    Returns:
        bool: True if the model has fallen, False otherwise.
    """
    check_series = pelvis_series[1:] # first frame is always 0 so can ignore
    for i, frame in enumerate(check_series):
        pelvis_ty = frame[1]
        if pelvis_ty < threshold:
            if debug:
                print(i, pelvis_ty, threshold, pelvis_ty < threshold)
            return True
    return False

def get_speed(pelvis_series, total_time, threshold=0.5):
    """
    Calculates the average speed of the model in the x-direction.

    - pelvis_series (list of list of float): Each element is a 6-element list of pelvis joint states.
    - total_time (float): Total time duration of the simulation.

    Returns:
        float: Average speed in the x-direction (pelvis_tx / total_time).
    """
    if len(pelvis_series) < 2 or total_time <= 0:
        return 0.0  # Not enough data or invalid time

    initial_tx = pelvis_series[0][2]
    final_tx = pelvis_series[-1][2]
    distance_traveled = final_tx - initial_tx
    speed = distance_traveled / total_time

    return speed, speed >= threshold 

def has_deviated(pelvis_series, threshold=0.5):
    """ 
    Calculates whether the pelvis deviates too far from the original straight desired path
    - pelvis_series: Each element is a 6-element list of pelvis joint states.
    """
    check_series = pelvis_series[1:] # first frame is always 0 so can ignore
    for i, frame in enumerate(check_series):
        pelvis_tz = frame[0]
        if abs(pelvis_tz) > threshold:
            return True
    return False

def has_rotated(pelvis_series, threshold):
    """ 
    Calculates whether the pelvis rotates too far from the original straight desired path
    - pelvis_series: Each element is a 6-element list of pelvis joint states.
    """
    check_series = pelvis_series[1:] # first frame is always 0 so can ignore
    for i, frame in enumerate(check_series):
        pelvis_qy = frame[4]
        if abs(pelvis_qy) > threshold:
            return True
    return False