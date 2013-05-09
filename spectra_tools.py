from functools import wraps
from numpy import exp
import numpy as np
import pylab as pl
import scipy

from constants import CM_FS_LINEAR
from utils import MetaArray, mean


def isotropic_average_2nd_order(func):
    """Function decorator to make function dependent on 2 lab-frame
    polarizations return its isotropic average instead."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return mean(func(*args, polarization=[p, p], **kwargs)
                    for p in 'xyz')
    return wrapper


def ensemble_average(hamiltonian, ensemble_size):
    """Function decorator to make function dependent on 2 lab-frame
    polarizations return its isotropic average instead."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return mean(func(hamiltonian.sample(), *args, **kwargs)
                        for _ in xrange(ensemble_size))
        return wrapper
    return decorator


def stft(x, fs, framesz, hop):
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w * x[i:i + framesamp])
                     for i in range(0, len(x) - framesamp, hopsamp)])
    return X


def wigner_distribution(pulse, t, freq_bounds=None, time_bounds=None):
    tau_range = t
    EE = np.array([pulse(t - tau / 2).conj() * pulse(t + tau / 2)
                   for tau in tau_range])
    W = meta_fft(MetaArray(EE, rw_freq=pulse.rw_default, ticks=tau_range),
                 axis=0, freq_bounds=freq_bounds, rev_freq=False,
                 positive_only=False)
    if time_bounds is not None:
        tau_keep = (tau_range > time_bounds[0]) & (tau_range < time_bounds[1])
    else:
        tau_keep = slice(None)
    return MetaArray(W[:, tau_keep], ticks=[tau_range[tau_keep], W.ticks])


def stft_spectrum(spectrum, framesize):
    pts = np.arange(framesize)
    w = exp(-(pts - np.mean(pts)) ** 2 / (2 * (float(framesize) / 6) ** 2))
    t = spectrum.ticks
    X = np.array([np.fft.fftshift(scipy.fft(w * spectrum[i:i + framesize],
                                            n=1024))
                  for i in range(0, len(spectrum) - framesize)])
    tau = t[int(framesize / 2):-int(framesize / 2)]
    omega = (np.fft.fftshift(np.fft.fftfreq(1024, (t[1] - t[0]) * (3e-5)))
             + spectrum.rw_freq)
    return MetaArray(X, ticks=[tau, omega])


def meta_fft(x, convert=CM_FS_LINEAR, axis=0, freq_bounds=None,
             rev_freq=True, positive_only=True):
    """
    Zero-pads a time-series starting at t=0 and returns the FFT. By default,
    the FFT is performed upon the first axis (not the last axis as defaults
    for np.fft.fft).

    First argument must be a MetaArray object wth ticks and rw_freq defined.

    Returns the FFT in another MetaArray object with ticks updated to the
    calculated frequencies (converted using the convert argument which defaults
    to cm to fs).
    """
    if positive_only:
        x_all = np.concatenate((np.zeros_like(x), x), axis=axis)
    else:
        x_all = x

    x_shifted = np.fft.fftshift(x_all, axes=axis)

    X = np.fft.fftshift(np.fft.fft(x_shifted, axis=axis), axes=axis)
    t = x.ticks
    dt = t[1] - t[0]

    rev = -1 if rev_freq else 1
    freqs = (rev * np.fft.fftshift(
        np.fft.fftfreq(x_all.shape[axis], dt * convert), axes=axis) + x.rw_freq)

    if freq_bounds is not None:
        i0 = np.argmin(np.abs(freqs - freq_bounds[0]))
        i1 = np.argmin(np.abs(freqs - freq_bounds[1]))
        X = X[min(i0, i1):(max(i0, i1) + 1)]
        freqs = freqs[min(i0, i1):(max(i0, i1) + 1)]

    metadata = x.metadata.copy()
    metadata.update(ticks=freqs[::rev])
    return MetaArray(X[::rev], **metadata)


def plot_pulse(pulse, fig=None):
    pl.figure(fig)

    pl.subplot(211)
    pl.cla()
    t = np.linspace(-1000, 1000, num=1000)
    pl.plot(t, 1e5 * pulse(t).real)
    pl.plot(t, 1e5 * pulse(t).imag)
    pl.plot(pulse.t_final * np.ones(2), [-100, 100], 'k--', linewidth=1)
    pl.legend(('Real part', 'Imag part'), prop={'size': 8})
    pl.ylabel('E(t) (arb units)')
    pl.ylim(-10, 10)

    pl.subplot(212)
    pl.cla()
    W = wigner_distribution(pulse, np.linspace(-5000, 5000, num=1000),
                            freq_bounds=(12000, 13000),
                            time_bounds=(-1000, 1000))
    pl.imshow(1e7 * W[::-1, :].real, aspect=.7,
              extent=(-1000, 1000, 12000, 13000))
    pl.plot(pulse.t_final * np.ones(2), [12000, 13000], 'k--', linewidth=1)
    pl.xlabel('Time (fs)')
    pl.ylabel('Frequency (1/cm)')


def plot_stft(pulse, framesize=150, fig=None):
    t = np.arange(-1000, 1000)
    X = stft_spectrum(MetaArray(pulse(t), ticks=t, rw_freq=pulse.rw_default),
                      framesize)
    pl.figure(fig)
    pl.cla()
    pl.contour(X.ticks[0], X.ticks[1], abs(X.T) ** 2, 20)
    pl.axis([-500, 500, 12000, 13000])


def linspan(A, B, n=100, debug=False):
    span = np.linspace(min((A.min(), B.min())), max((A.max(), B.max())), n)
    return span


def linspan_scale(A, B, n=100, debug=False):
    c_min = min((A.min(), B.min()))
    c_max = max((A.max(), B.max()))

    R = []
    if np.sign(A.max()) > 0 and np.sign(B.max()) > 0:
        R.append((A.max() / c_max, B.max() / c_max))
    if np.sign(A.min()) < 0 and np.sign(B.min()) < 0:
        R.append((A.min() / c_min, B.min() / c_min))

    if len(R) == 0:
        R1, R2 = (1, 1)
    elif len(R) == 1:
        R1, R2 = R[0]
    elif len(R) == 2:
        if np.prod(R[0]) > np.prod(R[1]):
            R1, R2 = R[0]
        else:
            R1, R2 = R[1]

    if debug:
        print A.min(), B.min()
        print A.max(), B.max()
        print c_min, c_max
        print R1, R2
    return np.linspace(c_min, c_max, n) * R1, np.linspace(c_min, c_max, n) * R2


def contourf_comp(X, Y, A, B, n=100, rel_scale=True, debug=False):
    pl.figure(figsize=(6, 3))
    if rel_scale:
        Z1, Z2 = linspan_scale(A, B, n, debug)
    else:
        Z1 = linspan(A, B, n, debug)
        Z2 = Z1

    pl.subplot(121)
    pl.contourf(X, Y, A, Z1)
    pl.subplot(122)
    pl.contourf(X, Y, B, Z2)
