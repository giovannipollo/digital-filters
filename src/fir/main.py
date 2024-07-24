import numpy as np
from scipy import signal
from fir_array.fir_array import FIRArray


def compute_impulse_response_coefficient(
    filter_order: int = 1, fs: float = 2, fc: float = 1
) -> np.array:
    """
    Compute the impulse response coefficients of a low-pass filter using a sinc function and a window hamming function.

    Args:
        filter_order (int): filter order, default is 1
        fs (float): sampling frequency, default is 2Hz
        fc (float): cutoff frequency, default is 1Hz

    Returns:
        h_w: np.array, impulse response coefficients
    """
    nyquist_frequency = fs / 2
    normalized_fc = (
        fc / nyquist_frequency
    )  # Normalize the cutoff frequency with respect to the Nyquist frequency

    # Ideal impulse response (sinc function)
    taps = np.arange(filter_order + 1)
    h = np.sinc(normalized_fc * (taps - filter_order / 2))

    # Hamming window
    w = np.hamming(filter_order + 1)

    # Apply window to the ideal impulse response
    h_w = h * w

    # Normalize the coefficients
    h_w /= np.sum(h_w)

    return h_w


if __name__ == "__main__":
    filter_order = 2  # Filter order
    fc = 0.1  # Cutoff frequency

    # Add fs as a parameter and compare it with scipy, by default is 2.
    h_w = compute_impulse_response_coefficient(filter_order=filter_order, fc=fc)
    print("Normalized coefficients computed by hand: ", h_w)
