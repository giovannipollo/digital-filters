import numpy as np
import matplotlib.pyplot as plt

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
        self.window_number = 0

    def adapt(self, x: np.ndarray, desired_signal: np.ndarray, plot_results: bool = True) -> tuple:
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

            # Calculate error signal
            error[i] = desired_signal[i] - output[i]

            # Update weights using the LMS rule
            self.weights += self.mu * error[i] * self.buffer
        
        if plot_results:
            self._plot_results(x, desired_signal, output, error)
            self.window_number += 1

        return output, error
    


    def _plot_results(self, input_signal: np.ndarray, desired_signal: np.ndarray, output_signal: np.ndarray, error: np.ndarray):
        """
        Plot the input signal, desired signal, filter output, and error signal.
        
        Args:
            input_signal (np.ndarray): The input signal array.
            desired_signal (np.ndarray): The desired signal array.
            output_signal (np.ndarray): The output of the filter.
            error (np.ndarray): The error signal.
        """
        plt.figure(figsize=(12, 10))

        plt.subplot(5, 1, 1)
        plt.plot(input_signal)
        plt.title('Input Signal')
        plt.ylabel('Amplitude')

        plt.subplot(5, 1, 2)
        plt.plot(desired_signal)
        plt.title('Desired Signal')
        plt.ylabel('Amplitude')

        plt.subplot(5, 1, 3)
        plt.plot(output_signal)
        plt.title('Filter Output')
        plt.ylabel('Amplitude')

        plt.subplot(5, 1, 4)
        plt.plot(error)
        plt.title('Error Signal')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        plt.subplot(5, 1, 5)
        plt.plot(desired_signal - output_signal)
        plt.title('Desired Signal - Filter Output')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig(f"plots/adaptive_filter/window_{self.window_number}_adaptive_filter.png")
        plt.close()

