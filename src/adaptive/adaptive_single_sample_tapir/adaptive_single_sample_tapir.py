import numpy as np

class AdaptiveFilterSingleSampleTapir:
    def __init__(self, num_taps: int, mu: float, num_channels: int = 3):
        """
        Initialize the multi-channel adaptive LMS filter.
        
        Args:
            num_taps (int): The number of filter taps (filter length)
            mu (float): The learning rate (step size)
            num_channels (int): Number of input channels (default 3 for accelerometer)
        """
        self.num_taps = num_taps
        self.mu = mu
        self.num_channels = num_channels
        
        # Initialize weights and buffers for each channel
        self.weights = np.zeros((num_channels, num_taps))
        self.buffer = np.zeros((num_channels, num_taps))
        
        # Add tracking for buffer filling
        self.buffer_filled = False
        self.samples_processed = 0

    def adapt(self, x: list, desired_signal: float):
        """
        Adapt the filter for each channel.
        
        Args:
            x (list): Current input samples [x, y, z] from accelerometer
            desired_signal (float): Current desired sample
        
        Returns:
            outputs (list): Filter outputs for each channel
            errors (list): Error signals for each channel
        """
        if len(x) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {len(x)}")

        outputs = np.zeros(self.num_channels)
        errors = np.zeros(self.num_channels)

        for channel in range(self.num_channels):
            # Shift buffer and add new sample for current channel
            self.buffer[channel, 1:] = self.buffer[channel, :-1]
            self.buffer[channel, 0] = x[channel]

        # Update samples counter and check if buffer is filled
        self.samples_processed += 1
        if self.samples_processed >= self.num_taps + 1:
            # The +1 is because of the implementation, but in normal condition it should not exist
            self.buffer_filled = True

        # Only perform computation if buffer is filled
        if self.buffer_filled:
            for channel in range(self.num_channels):
                # Compute output for current channel
                outputs[channel] = np.dot(self.weights[channel], self.buffer[channel])

                # Calculate error
                errors[channel] = desired_signal - outputs[channel]

                # Update weights for current channel
                self.weights[channel] += self.mu * errors[channel] * self.buffer[channel]

        return outputs.tolist()