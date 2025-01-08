import numpy as np


class AdaptiveFilter:
    def __init__(self) -> None:
        pass

    def adaptive_filter(
        self, input_signal, desired_signal, filter_order, learning_rate
    ):
        new_input_signal = np.concatenate(
            (np.zeros((filter_order, len(input_signal[0]))), input_signal)
        )
        num_channels = len(input_signal[0])
        num_samples = len(desired_signal)
        # Initialize filter coefficients for each channel
        weights = np.zeros((num_channels, filter_order))
        output_signal = np.zeros((num_samples, num_channels))
        error_signal = np.zeros((num_samples, num_channels))
        # Adaptive filtering using LMS
        for n in range(filter_order, len(new_input_signal)):
            # Input vector (most recent samples) for each channel
            x = new_input_signal[n - filter_order + 1 : n + 1, :]
            for channel in range(num_channels):
                # Reverse the order for dot product (most recent sample first)
                x_channel = x[::-1, channel]

                # print("Channel:", channel)
                # print("Input signal:", x_channel)
                # print("Desired signal:", desired_signal[n - filter_order])
                # Filter output for current channel
                output_signal[n - filter_order, channel] = np.dot(weights[channel], x_channel)

                # print("Output signal:", output_signal[n, channel])

                # Error signal for current channel
                error_signal[n - filter_order, channel] = desired_signal[n - filter_order] - output_signal[n - filter_order, channel]

                # print("Error signal:", error_signal[n, channel])
                # Update weights for current channel
                weights[channel] += learning_rate * error_signal[n - filter_order, channel] * x_channel

            #     print("Weights:", weights[channel])
            # print("-----------------")
        return output_signal
