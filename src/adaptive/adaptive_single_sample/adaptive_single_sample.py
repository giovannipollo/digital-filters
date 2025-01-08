import numpy as np

class AdaptiveFilterSingleSample:
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

            # print("Channel:", channel)
            # print("Buffer:", self.buffer[channel])
            # print("Desired signal:", desired_signal)
            # Compute output for current channel
            outputs[channel] = np.dot(self.weights[channel], self.buffer[channel])

            # print("Output signal:", outputs[channel])
            # Calculate error
            errors[channel] = desired_signal - outputs[channel]
            
            # print("Error signal:", errors[channel])
            # Update weights for current channel
            self.weights[channel] += self.mu * errors[channel] * self.buffer[channel]

        #     print("Weights:", self.weights[channel])
        # print("-----------------")
        return outputs.tolist()