import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import yaml
from scipy import signal
from iir_array.iir_array import IIRArray
from iir_single_sample.iir_single_sample import IIRSingleSample
from iir.iir_window_array.iir_window_array import IIRWindowArray
from utils.coefficient import split_iir_filter


def test_apply_iir_filter_array():
    # Load input signal
    with open("src/iir/test/input_signal.txt", "rb") as f:
        input_signal = np.loadtxt(f)

    # Load coefficients
    with open("src/iir/test/coefficient_4th_order.yaml") as f:
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


def test_apply_iir_filter_window_array():
    # Load input signal
    with open("src/iir/test/input_signal.txt", "rb") as f:
        input_signal = np.loadtxt(f)

    # Load coefficients
    with open("src/iir/test/coefficient_4th_order.yaml") as f:
        coefficient = yaml.safe_load(f)
    b = coefficient["b"]
    a = coefficient["a"]

    # Create IIRWindow instance and apply filter
    iir_window = IIRWindowArray(b=b, a=a)

    window_length_samples = 512
    window_shift_samples = 64
    y = np.zeros_like(input_signal)
    first_window = True
    for i in range(0, len(input_signal), window_shift_samples):
        if first_window:
            y[i : i + window_length_samples] = iir_window.apply_iir_filter(
                x=input_signal[i : i + window_length_samples]
            )
            first_window = False
        else:
            y[
                i
                + window_length_samples
                - window_shift_samples : i
                + window_length_samples
            ] = iir_window.apply_iir_filter(
                x=input_signal[
                    i
                    + window_length_samples
                    - window_shift_samples : i
                    + window_length_samples
                ]
            )

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
    with open("src/iir/test/coefficient_4th_order.yaml") as f:
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


def test_4th_order_coefficient():
    # Load coefficients
    with open("src/iir/test/coefficient_4th_order.yaml") as f:
        coefficient = yaml.safe_load(f)
    b = coefficient["b"]
    a = coefficient["a"]

    # Compute the coefficients with scipy
    b_scipy, a_scipy = signal.iirfilter(
        N=4, Wn=[0.4, 4], btype="band", ftype="butter", fs=64
    )

    # Compute the mean squared error
    mse_b = np.mean((b - b_scipy) ** 2)
    mse_a = np.mean((a - a_scipy) ** 2)

    assert mse_b < 1e-10
    assert mse_a < 1e-10


def test_double_2nd_order_filter():
    # Load coefficients
    with open("src/iir/test/coefficient_4th_order.yaml") as f:
        coefficient = yaml.safe_load(f)

    b = coefficient["b"]
    a = coefficient["a"]

    (b1, a1), (b2, a2) = split_iir_filter(b=b, a=a)

    # Load input signal
    with open("src/iir/test/input_signal.txt", "rb") as f:
        input_signal = np.loadtxt(f)

    y_scipy_intermediate = signal.lfilter(b=b1, a=a1, x=input_signal)
    y_scipy = signal.lfilter(b=b2, a=a2, x=y_scipy_intermediate)

    y_scipy_reference = signal.lfilter(b=b, a=a, x=input_signal)

    # Compute the mean squared error
    mse = np.mean((y_scipy - y_scipy_reference) ** 2)
    mae = np.mean(np.abs(y_scipy - y_scipy_reference))
    max_abs_diff = np.max(np.abs(y_scipy - y_scipy_reference))
    
    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-4

if __name__ == "__main__":
    pytest.main()
