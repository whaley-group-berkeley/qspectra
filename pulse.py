from abc import ABCMeta, abstractmethod
import numpy as np

from constants import GAUSSIAN_SD_FWHM

#i add a line here
class Pulse(object):
    """
    Abstract base class defining the Pulse API used by spectroscopy
    simulation methods

    Pulse instances are simply functions which return their complex-valued
    electric field as a function of time, with additional `t_init` and `t_final`
    attributes that give set times at which they start and end.

    Attributes
    ----------
    t_init : float
        Initital time for the pulse.
    t_final : float
        Final time for the pulse.
    """

    __meta__ = ABCMeta

    @abstractmethod
    def __call__(self, t, rw_freq):
        """
        Evaluate this pulse at the given time and frequency

        Parameters
        ----------
        t : number or np.ndarray
            Time at which to evaluate the pulse.
        rw_freq : number
            Rotating wave frequency (in same units as carrier_freq) at which
            to evaluate the pulse.

        Returns
        -------
        out : number or np.ndarray
            Complex-values of the pulse electric field at requested time t.
        """


class CustomPulse(Pulse):
    """
    Define a Pulse whose field is given by a custom function
    """
    def __init__(self, t_init, t_final, call):
        """
        Parameters
        ----------
        t_init : float
            Initital time for the pulse.
        t_final : float
            Final time for the pulse.
        call : function
            Function to call to return the pulse electric field at a given time
            and rotating wave frequency.
        """
        self.t_init = t_init
        self.t_final = t_final
        self.call = call

    def __call__(self, t, rw_freq):
        return self.call(t, rw_freq)


class GaussianPulse(Pulse):
    def __init__(self, carrier_freq, fwhm, t_peak=0, scale=1, freq_convert=1,
                 pulse_length=3):
        """
        Initialize a Gaussian pulse

        Parameters
        ----------
        carrier_freq : number
            Pulse carrier frequency (in frequency units).
        fwhm : number
            Pulse full-width-at-half-maximum (in time units).
        t_peak : number, optional
            Central peak time for this Gaussian pulse (default 0).
        scale : number, optional
            Scale factor which sets the maximum amplitude in the time domain
            (default 1).
        freq_convert : number, optional
            Conversion factor from frequency to time units. Default value is 1;
            set to constants.CM_FS to specify frequencies in angular cm^-1 and
            time in fs.
        t_limits_multiple : number, optional
            Number of standard deviations before and after t_peak to include in
            the pulse time limits (default 3).
        """
        sigma = GAUSSIAN_SD_FWHM * fwhm
        self.two_sigma_squared = 2 * sigma ** 2

        self.t_init = t_peak - 3 * sigma
        self.t_final = t_peak + 3 * sigma

        self.t_peak = t_peak
        self.carrier_freq = carrier_freq
        self.scale = scale
        self.freq_convert = freq_convert

    def __call__(self, t, rw_freq):
        return (self.scale *
                np.exp((1j * self.freq_convert * (self.carrier_freq - rw_freq)
                        * (t - self.t_peak))
                       - (t - self.t_peak) ** 2 / self.two_sigma_squared))
