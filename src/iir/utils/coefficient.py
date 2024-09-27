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
    Compute the impulse response coefficients of a low-pass filter using a sinc function and a window hamming function.

    Args:
        filter_order (int): filter order, default is 1
        fs (float): sampling frequency, default is 2Hz
        fc (float): cutoff frequency. If it is a list, it is a bandpass filter, default is 1Hz
        filter_type (str): filter type, default is 'butter'
        band_type (str): band type, default is 'low'

    Returns:
        tuple: impulse response coefficients in the form of a tuple (b, a)
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
        else:
            raise ValueError("band_type must be 'bandpass'") from None
    else:
        raise ValueError("filter_type must be 'butter'") from None

    return b, a
