import numpy as np

class WindowedAdaptiveFilter:
    def __init__(self, filter_order, learning_rate) -> None:
        self.prev_weights = None
        self.prev_input_buffer = None
        self.filter_order = filter_order
        self.learning_rate = learning_rate
        
    def process_window(self, input_signal: np.ndarray, desired_signal: np.ndarray):
        """
        Process signal in windows while maintaining filter state between windows.
        
        Args:
            input_signal: Input signal array of shape (samples, channels)
            desired_signal: Desired signal array of shape (samples,)
            filter_order: Length of the FIR filter
            learning_rate: Step size for weight updates
            overlap: Number of samples to overlap between windows
            
        Returns:
            output_signal: Filtered output signal
            error_signal: Error signal for the current window
        """
        num_channels = len(input_signal[0])
        num_samples = len(desired_signal)
        
        # Initialize or reuse weights from previous window
        if self.prev_weights is None:
            weights = np.zeros((num_channels, self.filter_order))
        else:
            weights = self.prev_weights.copy()
            
        # Initialize output and error signals
        output_signal = np.zeros((num_samples, num_channels))
        error_signal = np.zeros((num_samples, num_channels))
        subtract_idx = 0

        # If we have a previous input buffer, prepend it to current input
        if self.prev_input_buffer is not None:
            input_buffer = np.vstack((self.prev_input_buffer, input_signal))
            # Adjust starting index to account for buffer
            start_idx = self.filter_order
            subtract_idx = self.filter_order
            num_samples += self.filter_order
        else:
            input_buffer = input_signal
            start_idx = self.filter_order
            subtract_idx = 0
            
        # Adaptive filtering using LMS
        for n in range(start_idx, num_samples):
            # Input vector (most recent samples) for each channel
            x = input_buffer[n-self.filter_order+1:n+1, :]
            
            for channel in range(num_channels):
                # Reverse the order for dot product (most recent sample first)
                x_channel = x[::-1, channel]
                
                # Filter output for current channel
                output_signal[n - subtract_idx, channel] = np.dot(weights[channel], x_channel)

                # Error signal for current channel
                error_signal[n - subtract_idx, channel] = desired_signal[n - subtract_idx] - output_signal[n - subtract_idx, channel]
                
                # Update weights for current channel
                weights[channel] += self.learning_rate * error_signal[n - subtract_idx, channel] * x_channel
        
        # Store the last filter_order samples for the next window
        self.prev_input_buffer = input_signal[-self.filter_order:]
        # Store the final weights for the next window
        self.prev_weights = weights
        return output_signal

    def reset(self):
        """Reset the filter state."""
        self.prev_weights = None
        self.prev_input_buffer = None