import numpy as np

class IIRWindow:
    def __init__(self, b, a):
        self.b = np.asarray(b)
        self.a = np.asarray(a)
        self.taps = len(b)
        
        # Initialize state buffers for previous inputs and outputs
        self.x_history = np.zeros(self.taps - 1)  # Previous inputs
        self.y_history = np.zeros(self.taps - 1)  # Previous outputs
        
    def apply_iir_filter(self, x):
        """
        Apply IIR filter to input signal while maintaining state between calls.
        
        Args:
            x (array): Input signal window
        
        Returns:
            array: Filtered signal window
        """
        # Ensure input is a numpy array
        input_signal = np.asarray(x).flatten()
        
        if input_signal.ndim != 1 or self.b.ndim != 1 or self.a.ndim != 1:
            raise ValueError("Input signal, numerator and denominator coefficients must be 1D arrays")

        input_signal_length = len(input_signal)
        y = np.zeros(input_signal_length)
        
        # Create extended input signal with history
        x_extended = np.concatenate([self.x_history, input_signal])
        y_extended = np.concatenate([self.y_history, y])
        
        # Apply filter
        for n in range(input_signal_length):
            n_offset = n + len(self.x_history)  # Offset for extended arrays
            
            # Apply FIR part (b coefficients)
            for k in range(self.taps):
                y_extended[n_offset] += self.b[k] * x_extended[n_offset - k]
            
            # Apply IIR part (a coefficients)
            for k in range(1, self.taps):
                y_extended[n_offset] -= self.a[k] * y_extended[n_offset - k]
            
            y_extended[n_offset] /= self.a[0]
            y[n] = y_extended[n_offset]
        
        # Update state for next window
        self.x_history = x_extended[-self.taps+1:]
        self.y_history = y_extended[-self.taps+1:]
        
        return y
    
    def reset_state(self):
        """Reset the filter's state buffers to zeros."""
        self.x_history = np.zeros(self.taps - 1)
        self.y_history = np.zeros(self.taps - 1)