import numpy as np

class FIRArray:
    def __init__(self):
        pass
    
    @staticmethod
    def apply_fir_filter(x, h):
        """
        Apply FIR filter to input signal.
        
        Args:
            x (array): Input signal
            h (array): Filter coefficients
        
        Returns:
            array: Filtered signal
        """
        # Ensure input is a numpy array
        x = np.asarray(x)
        h = np.asarray(h)
        
        if x.ndim != 1 or h.ndim != 1:
            raise ValueError("Both input signal and filter coefficients must be 1D arrays")

        # Compute the number of taps
        taps = len(h)
        input_signal_length = len(x)
        y = np.zeros(input_signal_length)
        
        # Execute the convolution, which is the core of the FIR filter
        for n in range(input_signal_length):
            for k in range(taps):
                if n - k >= 0:
                    y[n] += h[k] * x[n - k]
        return y