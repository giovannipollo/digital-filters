import numpy as np

class IIRArray:
    def __init__(self, b, a):
        self.b = b
        self.a = a

    def apply_iir_filter(self, x):
        """
        Apply IIR filter to input signal.
        
        Args:
            x (array): Input signal
        
        Returns:
            array: Filtered signal
        """
        # Ensure input is a numpy array
        input_signal = np.asarray(x).flatten()
        b = np.asarray(self.b)
        a = np.asarray(self.a)
        
        if input_signal.ndim != 1 or b.ndim != 1 or a.ndim != 1:
            raise ValueError("Input signal, numerator and denominator coefficients must be 1D arrays")

        # Compute the number of taps
        taps = len(b)
        input_signal_length = len(input_signal)
        y = np.zeros(input_signal_length)
        
        for n in range(input_signal_length):
            for k in range(taps):
                if n - k >= 0:
                    y[n] = y[n] + b[k] * input_signal[n - k]
            for k in range(1, taps):
                if n - k >= 0:
                    y[n] = y[n] - a[k] * y[n - k]
            y[n] = y[n] / a[0]
        return y