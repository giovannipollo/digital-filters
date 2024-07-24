import numpy as np


class IIRSingleSample:
    def __init__(self, b: np.ndarray, a: np.ndarray, filter_order: int):
        self.b = b
        self.a = a

        if len(b) != filter_order + 1:
            raise ValueError("The length of the numerator coefficients (b) must be equal to the filter order + 1")
        if len(a) != filter_order + 1:
            raise ValueError("The length of the denominator coefficients (a) must be equal to the filter order + 1")
        if len(b) != len(a):
            raise ValueError("The length of the numerator and denominator coefficients (b and a respectively) must be equal")

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
            y = y + self.a[i] * self.output_buffer[i - 1]
        for i in range(1, len(self.b)):
            self.input_buffer[i] = self.input_buffer[i - 1]
        self.input_buffer[0] = x
        for i in range(1, len(self.a)):
            self.output_buffer[i] = self.output_buffer[i - 1]
        self.output_buffer[0] = y
        return y
