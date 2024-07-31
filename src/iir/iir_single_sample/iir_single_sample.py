import numpy as np


class IIRSingleSample:
    def __init__(self, b: np.ndarray, a: np.ndarray, filter_order: int):
        """
        Initialize IIR filter.

        Args:
            b (np.ndarray): Numerator coefficients
            a (np.ndarray): Denominator coefficients
            filter_order (int): Filter order 

        Returns:
            None

        Raises:
            ValueError: If the length of the numerator coefficients (b) is not equal to the filter order + 1
            ValueError: If the length of the denominator coefficients (a) is not equal to the filter order + 1
            ValueError: If the length of the numerator and denominator coefficients (b and a respectively) are not equal

        """
        self.b = b
        self.a = a

        if len(b) != filter_order + 1:
            raise ValueError("The length of the numerator coefficients (b) must be equal to the filter order + 1")
        if len(a) != filter_order + 1:
            raise ValueError("The length of the denominator coefficients (a) must be equal to the filter order + 1")
        if len(b) != len(a):
            raise ValueError("The length of the numerator and denominator coefficients (b and a respectively) must be equal")
        
        self.num_taps = filter_order + 1
        self.input_buffer = np.zeros(filter_order)
        self.output_buffer = np.zeros(filter_order)

    def apply_iir_filter(self, x: float) -> float:
        """
        Apply IIR filter to input signal.

        Args:
            x (float): Input sample

        Returns:
            float: Filtered sample
        """
        y = self.b[0] * x
        for i in range(1, len(self.b)):
            y = y + self.b[i] * self.input_buffer[i - 1]
        for i in range(1, len(self.a)):
            y = y - self.a[i] * self.output_buffer[i - 1]
        
        # Rotate the input buffer to the right of one position
        for i in range(1, len(self.b) - 1):
            self.input_buffer[self.num_taps - 1 - i] = self.input_buffer[self.num_taps - 2 - i]
        self.input_buffer[0] = x

        # Rotate the output buffer to the right of one position
        for i in range(1, len(self.a) - 1):
            self.output_buffer[self.num_taps - 1 - i] = self.output_buffer[self.num_taps - 2 -i]
        self.output_buffer[0] = y
        return y
