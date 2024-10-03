import numpy as np

class AdaptiveLMSFilterSingleSample:
    def __init__(self, num_taps: int, mu: float):
        """
        Initialize the adaptive LMS filter.
        
        Args:
            num_taps (int): The number of filter taps (filter length).
            mu (float): The learning rate (step size).
        """
        self.num_taps = num_taps
        self.mu = mu
        self.weights = np.zeros(num_taps)  # Initialize weights to zero
        self.buffer = np.zeros(num_taps)   # Initialize buffer to store the input signal

    def adapt(self, x: float, desired_signal: float):
        """
        Adapt the filter to minimize the error between the filter output and the desired signal.
        
        Args:
            x (float): The current input sample.
            desired_signal (float): The current desired sample (reference).
        
        Returns:
            output (float): The output of the filter.
            error (float): The error signal.
        """
        # Shift buffer and add new input sample
        self.buffer[1:] = self.buffer[:-1]  # Shift buffer to the right
        self.buffer[0] = x       # Insert the new sample at the beginning

        # Filter output (dot product of weights and buffer)
        output = np.dot(self.weights, self.buffer)

        # Calculate error (desired signal - filter output)
        error = desired_signal - output

        # Update weights using the LMS rule
        self.weights += self.mu * error * self.buffer

        return output, error
