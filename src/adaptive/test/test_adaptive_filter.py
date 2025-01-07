import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import yaml
from scipy import signal
from adaptive_filter_reference import AdaptiveFilter
from adaptive_filter_window.adaptive_filter_window import WindowedAdaptiveFilter
from adaptive_single_sample.adaptive_single_sample import AdaptiveFilterSingleSample
from adaptive.adaptive_single_sample_tapir.adaptive_single_sample_tapir import AdaptiveFilterSingleSampleTapir


def test_adaptive_filter_reference():
    # Load input signal
    input_signal = []
    with open("src/adaptive/test/input_signal.txt", "rb") as f:
        for line in f:
            # Convert bytes to string and clean
            line_str = line.decode("utf-8")
            clean_line = line_str.strip("[]").strip()
            # Remove the [ ]
            clean_line = clean_line.replace("[", "").replace("]", "")
            # Convert strings to floats, filtering empty strings
            numbers = [float(x) for x in clean_line.split() if x]
            input_signal.append(numbers)

    input_signal = np.array(input_signal)

    # Load desired signal
    with open("src/adaptive/test/desired_signal.txt", "rb") as f:
        desired_signal = np.loadtxt(f)

    # Load output signal
    with open("src/adaptive/test/output_signal.txt", "rb") as f:
        output_signal = np.loadtxt(f)

    AdaptiveFilterReference = AdaptiveFilter()
    filter_order = 50
    learning_rate = 8e-5
    y = AdaptiveFilterReference.adaptive_filter(
        input_signal, desired_signal, filter_order, learning_rate
    )

    # Do the mean over the channels of y
    y = (y[:, 0] + y[:, 1] + y[:, 2]) / 3

    # Check that y is equal to output_signal
    assert np.allclose(y, output_signal, atol=1e-5)


def test_adaptive_filter_windowed():
    # Load input signal
    input_signal = []
    with open("src/adaptive/test/input_signal.txt", "rb") as f:
        for line in f:
            # Convert bytes to string and clean
            line_str = line.decode("utf-8")
            clean_line = line_str.strip("[]").strip()
            # Remove the [ ]
            clean_line = clean_line.replace("[", "").replace("]", "")
            # Convert strings to floats, filtering empty strings
            numbers = [float(x) for x in clean_line.split() if x]
            input_signal.append(numbers)

    input_signal = np.array(input_signal)

    # Load desired signal
    with open("src/adaptive/test/desired_signal.txt", "rb") as f:
        desired_signal = np.loadtxt(f)

    # Load output signal
    with open("src/adaptive/test/output_signal.txt", "rb") as f:
        output_signal = np.loadtxt(f)

    AdaptiveFilterReference = AdaptiveFilter()
    filter_order = 50
    learning_rate = 8e-5
    y = AdaptiveFilterReference.adaptive_filter(
        input_signal=input_signal,
        desired_signal=desired_signal,
        filter_order=filter_order,
        learning_rate=learning_rate,
    )

    # Do the mean over the channels of y
    y = (y[:, 0] + y[:, 1] + y[:, 2]) / 3

    windowed_adaptive_filter = WindowedAdaptiveFilter(
        filter_order=50, learning_rate=8e-5
    )

    window_length_samples = 256
    window_shift_samples = 64
    y_windowed = np.zeros_like(input_signal)
    first_window = True
    for i in range(
        0, len(input_signal) - window_length_samples + 1, window_shift_samples
    ):
        if first_window:
            y_windowed[i : i + window_length_samples] = (
                windowed_adaptive_filter.process_window(
                    input_signal=input_signal[i : i + window_length_samples],
                    desired_signal=desired_signal[i : i + window_length_samples],
                )
            )
            first_window = False
        else:
            y_windowed[
                i
                + window_length_samples
                - window_shift_samples : i
                + window_length_samples
            ] = windowed_adaptive_filter.process_window(
                input_signal=input_signal[
                    i
                    + window_length_samples
                    - window_shift_samples : i
                    + window_length_samples
                ],
                desired_signal=desired_signal[
                    i
                    + window_length_samples
                    - window_shift_samples : i
                    + window_length_samples
                ],
            )

    # Do the mean over the channels of y_windowed
    y_windowed = (y_windowed[:, 0] + y_windowed[:, 1] + y_windowed[:, 2]) / 3

    # # Save y_windowed to check the output
    # with open("src/adaptive/test/y_windowed.txt", "wb") as f:
    #     np.savetxt(f, y_windowed)

    mse = np.mean((y_windowed - output_signal) ** 2)
    mae = np.mean(np.abs(y_windowed - output_signal))
    max_abs_diff = np.max(np.abs(y_windowed - output_signal))

    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-5

    mse = np.mean((y - y_windowed) ** 2)
    mae = np.mean(np.abs(y - y_windowed))
    max_abs_diff = np.max(np.abs(y - y_windowed))

    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-5


def test_adaptive_filter_single_sample():
    # Load input signal
    input_signal = []
    with open("src/adaptive/test/input_signal.txt", "rb") as f:
        for line in f:
            # Convert bytes to string and clean
            line_str = line.decode("utf-8")
            clean_line = line_str.strip("[]").strip()
            # Remove the [ ]
            clean_line = clean_line.replace("[", "").replace("]", "")
            # Convert strings to floats, filtering empty strings
            numbers = [float(x) for x in clean_line.split() if x]
            input_signal.append(numbers)

    input_signal = np.array(input_signal)

    # Load desired signal
    with open("src/adaptive/test/desired_signal.txt", "rb") as f:
        desired_signal = np.loadtxt(f)

    # Load output signal
    with open("src/adaptive/test/output_signal.txt", "rb") as f:
        output_signal = np.loadtxt(f)

    AdaptiveFilterReference = AdaptiveFilter()
    filter_order = 50
    learning_rate = 8e-5
    y = AdaptiveFilterReference.adaptive_filter(
        input_signal=input_signal,
        desired_signal=desired_signal,
        filter_order=filter_order,
        learning_rate=learning_rate,
    )

    # with open("src/adaptive/test/y.txt", "wb") as f:
    #     np.savetxt(f, y)
        
    # Do the mean over the channels of y
    y = (y[:, 0] + y[:, 1] + y[:, 2]) / 3

    adaptive_filter_single_sample = AdaptiveFilterSingleSample(
        num_taps=50, mu=8e-5, num_channels=3
    )

    y_single_sample = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        y_single_sample[i] = adaptive_filter_single_sample.adapt(
            x=input_signal[i], desired_signal=desired_signal[i]
        )

    # Do the mean over the channels of y_single_sample
    y_single_sample = (y_single_sample[:, 0] + y_single_sample[:, 1] + y_single_sample[:, 2]) / 3

    mse = np.mean((y_single_sample - output_signal) ** 2)
    mae = np.mean(np.abs(y_single_sample - output_signal))
    max_abs_diff = np.max(np.abs(y_single_sample - output_signal))
    
    assert mse < 1e-2
    assert mae < 1e-2
    assert max_abs_diff < 1e-1


def test_adaptive_filter_single_sample_tapir():
    # Load input signal
    input_signal = []
    with open("src/adaptive/test/input_signal.txt", "rb") as f:
        for line in f:
            # Convert bytes to string and clean
            line_str = line.decode("utf-8")
            clean_line = line_str.strip("[]").strip()
            # Remove the [ ]
            clean_line = clean_line.replace("[", "").replace("]", "")
            # Convert strings to floats, filtering empty strings
            numbers = [float(x) for x in clean_line.split() if x]
            input_signal.append(numbers)

    input_signal = np.array(input_signal)

    # Load desired signal
    with open("src/adaptive/test/desired_signal.txt", "rb") as f:
        desired_signal = np.loadtxt(f)

    # Load output signal
    with open("src/adaptive/test/output_signal.txt", "rb") as f:
        output_signal = np.loadtxt(f)

    AdaptiveFilterReference = AdaptiveFilter()
    filter_order = 50
    learning_rate = 8e-5
    y = AdaptiveFilterReference.adaptive_filter(
        input_signal=input_signal,
        desired_signal=desired_signal,
        filter_order=filter_order,
        learning_rate=learning_rate,
    )

    with open("src/adaptive/test/y.txt", "wb") as f:
        np.savetxt(f, y)
        
    # Do the mean over the channels of y
    y = (y[:, 0] + y[:, 1] + y[:, 2]) / 3

    adaptive_filter_single_sample = AdaptiveFilterSingleSampleTapir(
        num_taps=50, mu=8e-5, num_channels=3
    )

    y_single_sample = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        y_single_sample[i] = adaptive_filter_single_sample.adapt(
            x=input_signal[i], desired_signal=desired_signal[i]
        )

    # Do the mean over the channels of y_single_sample
    y_single_sample = (y_single_sample[:, 0] + y_single_sample[:, 1] + y_single_sample[:, 2]) / 3

    mse = np.mean((y_single_sample - output_signal) ** 2)
    mae = np.mean(np.abs(y_single_sample - output_signal))
    max_abs_diff = np.max(np.abs(y_single_sample - output_signal))
    
    assert mse < 1e-10
    assert mae < 1e-5
    assert max_abs_diff < 1e-5


if __name__ == "__main__":
    test_adaptive_filter_single_sample_tapir()
    # pytest.main()
