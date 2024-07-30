from utils.coefficient import compute_impulse_response_coefficient
from iir_array.iir_array import IIRArray
import numpy as np

if __name__ == "__main__":
    filter_order = 4  # Filter order
    fc = [0.4, 4]  # Cutoff frequency
    fs = 64  # Sampling frequency

    b, a = compute_impulse_response_coefficient(
        filter_order=filter_order, fc=fc, band_type="bandpass", fs=fs
    )
    print("b: ", b)
    print("a: ", a)
