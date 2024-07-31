import numpy as np


class FIRSingleSample:
    def __init__(self, filter_order: int, coefficients: np.ndarray):
        """
        Initialize FIR filter with filter order and coefficients.

        Args:
            filter_order (int): Filter order
            coefficients (array): Filter coefficients

        Returns:
            None
        """
        self.filter_order = filter_order
        self.number_of_taps = filter_order + 1
        self.coefficients = coefficients
        self.input_buffer = np.zeros(filter_order)

    def apply_fir_filter(self, x: float) -> float:
        """
        Apply FIR filter to input signal.

        Args:
            x (float): Input sample

        Returns:
            float: Filtered sample
        """
        out_filtered = x * self.coefficients[0]
        for i in range(1, self.number_of_taps):
            out_filtered = (
                out_filtered + self.input_buffer[i - 1] * self.coefficients[i]
            )

        # Rotate the input buffer to the right of one position
        for i in range(1, self.number_of_taps - 1):
            self.input_buffer[self.number_of_taps - 1 - i] = self.input_buffer[
                self.number_of_taps - 2 - i
            ]
        self.input_buffer[0] = x
