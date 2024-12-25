import numpy as np
from scipy import signal
from typing import Union, Literal


def compute_impulse_response_coefficient(
    filter_order: int = 1,
    fs: float = 2,
    fc: Union[float, list[float]] = 1,
    filter_type: str = "butter",
    band_type: Literal["low", "high", "band", "bandpass", "bandstop"] = "bandpass",
) -> tuple[list, list]:
    """
    Compute impulse response coefficients for digital Butterworth filters.

    This function wraps scipy.signal.butter to generate filter coefficients for
    various filter configurations. It includes input validation and proper error
    handling for different band types.

    Args:
        filter_order (int): Order of the filter. Higher orders give sharper frequency cutoffs.
        Defaults to 1.
        fs (float): Sampling frequency in Hz. Defaults to 2Hz.
        fc (float | list[float]): Cutoff frequency in Hz. For bandpass filters, provide
        [low_freq, high_freq]. For low/high pass, provide single frequency.
        Defaults to 1Hz.
        filter_type (str): Type of filter to design. Currently only supports 'butter'
        for Butterworth filters.
        band_type (str): Filter response type:
            - 'low': lowpass filter
            - 'high': highpass filter
            - 'bandpass': bandpass filter (requires fc as [low, high])
        Defaults to 'bandpass'.

    Returns:
        tuple[list, list]: Filter coefficients (b, a) where:
            - b: numerator coefficients
            - a: denominator coefficients

    Raises:
        ValueError: If band_type/filter_type combination is invalid or if fc format
        doesn't match the band_type requirements.

    Example:
        >>> # Create a lowpass Butterworth filter at 30Hz, sampling at 1kHz
        >>> b, a = compute_impulse_response_coefficient(
        ...     filter_order=4,
        ...     fs=1000,
        ...     fc=30,
        ...     band_type='low'
        ... )
        >>> # Create a bandpass filter between 20-50Hz
        >>> b, a = compute_impulse_response_coefficient(
        ...     filter_order=4,
        ...     fs=1000,
        ...     fc=[20, 50],
        ...     band_type='bandpass'
        ... )
    """
    if filter_type == "butter":
        if band_type == "bandpass":
            try:
                b, a = signal.butter(N=filter_order, Wn=fc, btype=band_type, fs=fs)
            except Exception as e:
                if band_type == "bandpass" and not isinstance(fc, list):
                    raise ValueError(
                        "fc must be a list when band_type is 'bandpass'"
                    ) from e
                else:
                    raise e
        elif band_type == "low" or band_type == "high":
            try:
                b, a = signal.butter(N=filter_order, Wn=fc, btype=band_type, fs=fs)
            except Exception as e:
                if band_type == "low" or band_type == "high" and isinstance(fc, list):
                    raise ValueError(
                        "fc must be a float when band_type is 'low' or 'high'"
                    ) from e
                else:
                    raise e
        else:
            raise ValueError("band_type must be 'bandpass'") from None
    else:
        raise ValueError("filter_type must be 'butter'") from None

    return b, a


def split_iir_filter(b: np.ndarray, a: np.ndarray) -> tuple:
    """
    Split a 4th-order IIR filter into two 2nd-order filters.

    This function finds the zeros (z), poles (p), and gain (k) of the original
    4th-order filter using scipy.signal.tf2zpk, splits those roots evenly into
    two separate filters, and then converts them back to polynomial coefficients
    with scipy.signal.zpk2tf.

    Parameters:
    -----------
    b : np.ndarray
        The numerator coefficients of the 4th-order filter.
    a : np.ndarray
        The denominator coefficients of the 4th-order filter.

    Returns:
    --------
    (b1, a1), (b2, a2) : tuple
        Two pairs of 2nd-order filter coefficients. Each pair consists of a
        1D array of numerator coefficients and a 1D array of denominator
        coefficients.

    Notes:
    ------
    - The order of the split is assumed to be even (4th-order), dividing
      the zeros and poles into two sets of equal size.
    - The overall gain factor is split evenly by taking sqrt(k).

    Example:
    --------
    >>> import numpy as np
    >>> from scipy import signal
    >>> b = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    >>> a = np.array([1.0, -0.4, 0.2, -0.1, 0.05])
    >>> (b1, a1), (b2, a2) = split_iir_filter(b, a)
    >>> print(b1, a1)
    >>> print(b2, a2)
    """
    # Get the roots of the numerator and denominator
    z, p, k = signal.tf2zpk(b, a)
    
    # Split roots into two groups for 2nd-order filters
    z1, z2 = np.split(z, 2)
    p1, p2 = np.split(p, 2)
    
    # Convert back to polynomial coefficients
    b1, a1 = signal.zpk2tf(z1, p1, np.sqrt(k))  # Split the gain equally
    b2, a2 = signal.zpk2tf(z2, p2, np.sqrt(k))
    
    return (b1, a1), (b2, a2)