import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import yaml
from iir_array.iir_array import IIRArray
from scipy import signal
def test_apply_iir_filter():
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
    y = iir_array.apply_iir_filter(x=input_signal, b=b, a=a)

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