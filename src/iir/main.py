import numpy as np

def compute_impulse_response_coefficient(filter_order: int, fc: float) -> np.array:
    """
    Compute the impulse response coefficients of a butterworth filter

    Args:
        filter_order (int): filter order
        fc (float): cutoff frequency (normalized to Nyquist frequency). You should pass the cutoff frequency divided by 2 with respect to the cutoff frequency in Hz.

    Returns:
        h_w: np.array, impulse response coefficients
    """
    

if __name__ == "__main__":