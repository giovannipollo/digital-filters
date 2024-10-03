import numpy as np

class AdaptiveLMSFilterArray:
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

    def adapt(self, x: np.ndarray, desired_signal: np.ndarray):
        """
        Adapt the filter to minimize the error between the filter output and the desired signal.
        
        Args:
            x (np.ndarray): The input signal array.
            desired_signal (np.ndarray): The desired signal array (reference).
        
        Returns:
            output (np.ndarray): The output of the filter.

            error (np.ndarray): The error signal.
        """
        # Ensure input arrays have the same length
        assert len(x) == len(desired_signal), "Input and desired signal must have the same length"

        output = np.zeros(len(x))
        error = np.zeros(len(x))

        for i in range(len(x)):
            # Shift buffer and add new input sample
            self.buffer[1:] = self.buffer[:-1]  # Shift buffer to the right
            self.buffer[0] = x[i]  # Insert the new sample at the beginning

            # Filter output (dot product of weights and buffer)
            output[i] = np.dot(self.weights, self.buffer)

            # Calculate error (desired signal - filter output)
            error[i] = desired_signal[i] - output[i]

            # Update weights using the LMS rule
            self.weights += self.mu * error[i] * self.buffer

        return output, error