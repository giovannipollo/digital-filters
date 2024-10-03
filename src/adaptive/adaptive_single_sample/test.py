import numpy as np
import matplotlib.pyplot as plt
from adaptive_array import AdaptiveLMSFilterArray

# Generate sine and cosine waves
def generate_sine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def generate_cosine_wave(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.cos(2 * np.pi * freq * t)

# Parameters
duration = 1  # seconds
sample_rate = 1000  # Hz
freq = 10  # Hz
noise_amplitude = 0.2

# Generate signals
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
input_signal = generate_cosine_wave(freq, duration, sample_rate)
desired_signal = generate_sine_wave(freq, duration, sample_rate)

# Initialize and apply the LMS filter
num_taps = 64  # Increased number of taps to better handle the phase difference
mu = 0.01
lms_filter = AdaptiveLMSFilterArray(num_taps, mu)
output_signal, error = lms_filter.adapt(input_signal, desired_signal)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t, input_signal)
plt.title('Input Signal')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(t, desired_signal)
plt.title('Desired Signal (Sine)')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.plot(t, output_signal)
plt.title('LMS Filter Output')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 4)
plt.plot(t, error)
plt.title('Error Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.savefig('lms_filter_cosine_sine_signals.png')
plt.close()

# Calculate and print the Mean Squared Error (MSE)
mse = np.mean(error**2)
print(f"Mean Squared Error: {mse:.6f}")


print("All plots have been saved as PNG files.")