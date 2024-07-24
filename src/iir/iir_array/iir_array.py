import numpy as np

class IIRArray:
    def __init__(self, b, a):
        self.b = b
        self.a = a

    @staticmethod
    def apply_iir_filter(x, b, a):
        """
        Apply IIR filter to input signal.
        
        Args:
            x (array): Input signal
            b (array): Numerator coefficients
            a (array): Denominator coefficients
        
        Returns:
            array: Filtered signal
        """
        # Ensure input is a numpy array
        x = np.asarray(x)
        b = np.asarray(b)
        a = np.asarray(a)
        
        if x.ndim != 1 or b.ndim != 1 or a.ndim != 1:
            raise ValueError("Input signal, numerator and denominator coefficients must be 1D arrays")

        # Compute the number of taps
        taps = len(b)
        input_signal_length = len(x)
        y = np.zeros(input_signal_length)
        
        for n in range(input_signal_length):
            for k in range(taps):
                if n - k >= 0:
                    y[n] += b[k] * x[n - k]
            for k in range(1, taps):
                if n - k >= 0:
                    y[n] += a[k] * y[n - k]
        return y