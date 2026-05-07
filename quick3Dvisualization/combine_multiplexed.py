import numpy as np

def combine_multiplexed(multiplexed_data):
    """
    Combine 1024x16x4 multiplexed records into full 1024x61 data.

    Parameters
    ----------
    multiplexed_data : numpy.ndarray
        Expected shape: (num_depths, 16, 4)

    Returns
    -------
    full_data : numpy.ndarray
        Shape: (num_depths, 61)
    """

    # Validate shape
    if multiplexed_data.ndim != 3:
        raise ValueError("Input must be a 3D array")

    num_depths, num_records, num_sections = multiplexed_data.shape

    if num_records != 16 or num_sections != 4:
        raise ValueError("Incorrect input size. Expected shape (N, 16, 4)")

    # Allocate output array
    full_data = np.zeros((num_depths, 61))

    # Transfer data
    for rec in range(16):          # 0–15
        for sec in range(4):       # 0–3
            ch = sec * 16 + rec    # channel index (0-based now)
            if ch < 61:            # only 61 channels
                full_data[:, ch] = multiplexed_data[:, rec, sec]

    return full_data