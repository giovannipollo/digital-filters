# FIR Filter Implementation

This repository contains an implementation from scratch of a FIR filter.

## FIR Filter

A Finite Impulse Response (FIR) filter is a filter whose Impulse Response is of finite duration. For a FIR filter of order N, each value of the output sequence is a weighted sum of the most recent input values. This is clearly visible in the following picture, taken from the Wikipedia article of the FIR filter [https://en.wikipedia.org/wiki/Finite_impulse_response](https://en.wikipedia.org/wiki/Finite_impulse_response). 

![Wikipedia FIR filter](images/fir.png)

Mathematically, this can be expressed as follows:

$$
y[n] = b_0x[n] + b_1x[n-1] + b_2x[n-2] + \dots + b_Nx[n-N]
$$

This is equal to:

$$
\sum_{i=0}^{N} b_i x[n-i]
$$

where:

- $x[n]$ is the input signal at time $n$
- $y[n]$ is the output signal at time $n$
- $N$ is the order of the filter
- $b_i$ is the value of the impulse response at time $i$

> [!CAUTION]
> The relation between the filter order and the taps is the following:
> $$
> filter\_order = taps - 1
>$$
> So, for example, a filter of 5th order has 6 taps

