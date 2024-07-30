import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import time

def lfilter(b, a, x):
    a = np.array(a)
    b = np.array(b)
    x = np.array(x)
    y = np.zeros(len(x))
    for n in range(len(x)):
        y[n] = b[0] * x[n]
        for i in range(1, len(b)):
            if n - i >= 0:
                y[n] += b[i] * x[n - i]
        for j in range(1, len(a)):
            if n - j >= 0:
                y[n] -= a[j] * y[n - j]
        y[n] /= a[0]
    return y

def custom_filtfilt(b, a, x):
    y = lfilter(b, a, x)
    y = lfilter(b, a, y[::-1])
    return y[::-1]

# Generate a sample signal
t = np.linspace(0, 1, 10000, endpoint=False)
x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + np.random.normal(0, 0.1, len(t))

# Define filter coefficients (4th order Butterworth lowpass filter with cutoff at 15 Hz)
b, a = signal.butter(4, 15, fs=len(t), btype='low')

# Apply custom filtfilt
start_time = time.time()
y_custom = custom_filtfilt(b, a, x)
custom_time = time.time() - start_time

# Apply SciPy filtfilt
start_time = time.time()
y_scipy = signal.filtfilt(b, a, x)
scipy_time = time.time() - start_time

# Compute difference
diff = y_custom - y_scipy

# Plot results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.plot(t, y_custom, label='Custom filtfilt')
plt.plot(t, y_scipy, label='SciPy filtfilt')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal Comparison')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, diff)
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between Custom and SciPy filtfilt')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.semilogy(np.abs(np.fft.fftfreq(len(t), t[1]-t[0])), np.abs(np.fft.fft(x)), label='Original')
plt.semilogy(np.abs(np.fft.fftfreq(len(t), t[1]-t[0])), np.abs(np.fft.fft(y_custom)), label='Custom')
plt.semilogy(np.abs(np.fft.fftfreq(len(t), t[1]-t[0])), np.abs(np.fft.fft(y_scipy)), label='SciPy')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Response')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Custom filtfilt execution time: {custom_time:.6f} seconds")
print(f"SciPy filtfilt execution time: {scipy_time:.6f} seconds")
print(f"Maximum absolute difference: {np.max(np.abs(diff)):.6e}")
print(f"Mean absolute difference: {np.mean(np.abs(diff)):.6e}")