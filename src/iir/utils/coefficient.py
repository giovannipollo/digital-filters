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
