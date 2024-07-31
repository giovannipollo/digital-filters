import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import yaml
from scipy import signal
from iir_array.iir_array import IIRArray
from iir_single_sample.iir_single_sample import IIRSingleSample

def test_apply_iir_filter_array():
    # Load input signal
    with open("src/iir/test/input_signal.txt", "rb") as f:
        input_signal = np.loadtxt(f)

    # Load coefficients
    with open("src/iir/test/coefficient.yaml") as f:
        coefficient = yaml.safe_load(f)
    b = coefficient["b"]
    a = coefficient["a"]

    # Create IIRArray instance and apply filter
    iir_array = IIRArray(b=b, a=a)
    y = iir_array.apply_iir_filter(x=input_signal)

    # Compute the expected output signal with scipy
    y_scipy = signal.lfilter(b=b, a=a, x=input_signal)

    # Compute the mean squared error
    mse = np.mean((y - y_scipy) ** 2)
    mae = np.mean(np.abs(y - y_scipy))
    max_abs_diff = np.max(np.abs(y - y_scipy))

    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-5

def test_apply_iir_filter_single_sample():
    # Load input signal
    with open("src/iir/test/input_signal.txt", "rb") as f:
        input_signal = np.loadtxt(f)

    # Load coefficients
    with open("src/iir/test/coefficient.yaml") as f:
        coefficient = yaml.safe_load(f)
    b = coefficient["b"]
    a = coefficient["a"]

    # Create IIRSingleSample instance and apply filter
    iir_single_sample = IIRSingleSample(b=b, a=a, filter_order=len(b) - 1)
    y = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        y[i] = iir_single_sample.apply_iir_filter(x=input_signal[i])

    # Compute the expected output signal with scipy
    y_scipy = signal.lfilter(b=b, a=a, x=input_signal)

    # Compute the mean squared error
    mse = np.mean((y - y_scipy) ** 2)
    mae = np.mean(np.abs(y - y_scipy))
    max_abs_diff = np.max(np.abs(y - y_scipy))

    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-5

if __name__ == '__main__':
    pytest.main()