import numpy as np

class AdaptiveFilter:
    def __init__(self) -> None:
        pass

    
    def adaptive_filter(self, input_signal, desired_signal, filter_order, learning_rate):
        num_channels = len(input_signal[0])
        num_samples = len(desired_signal)
        # Initialize filter coefficients for each channel
        weights = np.zeros((num_channels, filter_order))
        output_signal = np.zeros((num_samples, num_channels))
        error_signal = np.zeros((num_samples, num_channels))
        # Adaptive filtering using LMS
        for n in range(filter_order, num_samples):
            # Input vector (most recent samples) for each channel
            x = input_signal[n-filter_order+1:n+1, :]
            for channel in range(num_channels):
                # Reverse the order for dot product (most recent sample first)
                x_channel = x[::-1, channel]
                
                # Filter output for current channel
                output_signal[n, channel] = np.dot(weights[channel], x_channel)
                
                # Error signal for current channel
                error_signal[n, channel] = desired_signal[n] - output_signal[n, channel]
                
                # Update weights for current channel
                weights[channel] += learning_rate * error_signal[n, channel] * x_channel
        return output_signal