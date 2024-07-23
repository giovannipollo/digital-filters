import numpy as np
from scipy import signal
from fir_array.fir_array import FIRArray

def compute_impulse_response_coefficient(filter_order, fc):
    """
    Compute the impulse response coefficients of a low-pass filter using a sinc function and a window hamming function.

    Args:
        filter_order (int): filter order
        fc (float): cutoff frequency (normalized to Nyquist frequency). You should pass the cutoff frequency divided by 2 with respect to the cutoff frequency in Hz.

    Returns:
        h_w: np.array, impulse response coefficients
    """
    # Ideal impulse response (sinc function)
    taps = np.arange(filter_order + 1)
    h = np.sinc(2 * fc * (taps - filter_order / 2))

    # Hamming window
    w = np.hamming(filter_order + 1)

    # Apply window to the ideal impulse response
    h_w = h * w

    # Normalize the coefficients
    h_w /= np.sum(h_w)

    return h_w

def fir_filter_array(x, h):
    """
    Apply FIR filter to input signal.
    
    Args:
        x (array): Input signal
        h (array): Filter coefficients
    
    Returns:
        array: Filtered signal
    """
    
    taps = len(h)
    input_signal_length = len(x)
    y = np.zeros(input_signal_length)
    
    for n in range(input_signal_length):
        for k in range(taps):
            if n - k >= 0:
                y[n] += h[k] * x[n - k]
    return y

def fir_filter_sample_by_sample(x, h):
    
if __name__ == "__main__":
    filter_order = 2  # Filter order
    fc = 0.05  # Cutoff frequency (normalized to Nyquist frequency, meaning it's divided by 2 with respect to the cutoff frequency in Hz). For this reason, in the sinc we have to multiply by 2

    h_w = compute_impulse_response_coefficient(filter_order, fc)
    print("Normalized coefficients computed by hand: ", h_w)




