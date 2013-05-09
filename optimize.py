from functools import wraps

from numpy import exp, pi, cos, sin, log
import numpy as np
import scipy.linalg
import scipy.optimize

from simulate import simulate_pump
from pulse import ShapedPulse, SumOfGaussiansPulse, CrabPulse
from operator_tools import (normalized_exclude_ground, meta_series_op,
                            exclude_ground, den_to_vec)
from spectra_tools import isotropic_average_2nd_order, mean
from utils import MetaArray, Registry


class PulseRegistry(Registry):
    def __call__(self, func):
        @wraps(func)
        def wrapper(**kwargs):
            return lambda x: func(x, **kwargs)
        return super(PulseRegistry, self).__call__(wrapper)


pulse_makers = PulseRegistry()


@pulse_makers
def crab_fixed_harmonics(args, n_harmonics, duration, fixed_time=250):
    N = n_harmonics
    freqs = (pi / fixed_time * np.linspace(-N, N, 2 * N + 1))
    return CrabPulse(args, duration, crab_freqs=freqs)


@pulse_makers
def crab_fixed_harmonics_var_duration(args, n_harmonics, fixed_time=250):
    duration = max(min(75 * args[0] + 225, 10), 1000)
    N = n_harmonics
    freqs = (pi / fixed_time * np.linspace(-N, N, 2 * N + 1))
    return CrabPulse(args[1:], duration, crab_freqs=freqs)



def pair_of_gaussians(args):
    weights = (np.array([12000, 12000, 55, 55, -300, 0]) +
               np.array(args) * np.array([800, 800, 400, 400, 600, 2*pi]))
    pulse = SumOfGaussiansPulse(np.array([.5, .5]), weights[0:2],
                                weights[2:4], np.array([0, weights[4]]),
                                np.array([0, weights[5]]),
                                suppress_errors=True)
    return pulse


def n_flexible_gaussians(args):
    args = np.array(args)

    n = int((len(args) + 3.)/5)
    w = np.r_[args[:(n-1)], 1 - sum(args[:(n-1)])/(n-1)]
    cf = 12000 + 900 * args[(n-1):(2*n-1)]
    fwhms = 55 * 10**args[(2*n-1):(3*n-1)]
    times = -300 + 600 * np.r_[0.5, args[(3*n-1):(4*n-2)]]
    phases = 2 * pi * np.r_[0, args[(4*n-2):(5*n-3)]]

    pulse = SumOfGaussiansPulse(w, cf, fwhms, times, phases,
                                suppress_errors=True)
    return pulse


@pulse_makers
def poly_pulse(args, flatten=False, **pulse_args):
    args = np.array(args)
    if not flatten:
        sf = lambda x: exp(1j*np.sum(args*x**np.arange(2, len(args)+2)))
    else:
        sf = lambda x: exp(1j*(np.sum(args*x**np.arange(2, len(args)+2))
                        - 1j*min(log(.5/exp(-x**2/2)),0)))
    return ShapedPulse(shape_func=sf, suppress_errors=True, **pulse_args)


def poly_phase_fourier_amp_pulse(args, amp_min=0.5, **pulse_args):
    phase_args = args[:10]
    phase_args_scaled = (np.array(phase_args)
                         * np.array([1./((n+2)*2**(n+1)) for n in range(10)]))

    amp_args = args[10:]
    if len(amp_args) % 2 != 1:
        raise Exception('wrong number of pulse args')
    a0 = amp_args[0]
    a = amp_args[1::2]
    b = amp_args[2::2]

    sf = lambda x: (exp(1j*np.sum(phase_args_scaled*x**np.arange(2, 10+2)))
                    * max(min(a0 + np.sum([a*cos(n*pi*x/3) + b*sin(n*pi*x/3)
                                           for n in range(1,len(a)+1)]),
                              1), amp_min))

    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


def gaussian_fourier_amp_pulse(args, amp_min=0.5, **pulse_args):

    pts = np.linspace(-3, 3, len(args) + 2)[1:-1]
    diff = pts[1] - pts[0]

    sf = lambda x: (max(min(np.sum(args*exp(-(x - pts)**2/(diff**2/2))),
                              1), amp_min))

    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


def gaussian_amp_poly_phase_pulse(args, amp_min=0.5, **pulse_args):

    poly_args = args[:3]
    amp_args = args[3:]

    pts = np.linspace(-3, 3, len(amp_args))
    diff = pts[1] - pts[0]

    sf = lambda x: (exp(1j*np.sum(poly_args*x**np.arange(2, len(poly_args)+2)))*
                    (max(min(np.sum(amp_args*exp(-(x - pts)**2/(diff**2/2))), 1),
                         amp_min)))

    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


def poly_fourier_amp_pulse(args, amp_min=0.5, **pulse_args):

    amp_args = args
    if len(amp_args) % 2 != 1:
        raise Exception('wrong number of pulse args')
    a0 = amp_args[0]
    a = amp_args[1::2]
    b = amp_args[2::2]

    sf = lambda x: (max(min(a0 + np.sum([a * cos(n * pi * x / 3)
                            + b * sin(n * pi * x / 3)
                            for n in range(1, len(a) + 1)]), 1), amp_min))

    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


@pulse_makers
def poly_pulse_scaled(args, flatten=False, **pulse_args):
    args = (np.array(args) * np.array([1. / ((n + 2) * 2 ** (n + 1))
                                       for n in range(len(args))]))
    if not flatten:
        sf = lambda x: exp(1j * np.sum(args * x ** np.arange(2, len(args) + 2)))
    else:
        sf = lambda x: exp(1j * (np.sum(args * x ** np.arange(2, len(args) + 2))
                           - 1j * min(log(.5 / exp(-x ** 2 / 2)), 0)))
    return ShapedPulse(shape_func=sf, suppress_errors=True, carrier_freq=12422,
                       **pulse_args)


# @pulse_makers
# def poly_pulse_fixed(args):
#     args = np.array(args)
#     a, b, c = args[:3]
#     sf = lambda x: exp(1j*(10*pi*(2*a-1)*x**2 + 20*pi*(2*b-1)*x**3 + 30*pi*(2*c-1)*x**3))
# #    td = args[3]*1000
#     return ShapedPulse(t_delay=td, shape_func=sf, suppress_errors=True)


@pulse_makers
def fixed_gaussians(args):
    w = args[0:12]
    cf = np.arange(12150, 12750, 50)
    fwhms = 250 * np.ones(12)
    times = 300 * np.r_[0, args[12:23]]
    phases = 2 * pi * np.r_[0, args[23:34]]

    pulse = SumOfGaussiansPulse(w, cf, fwhms, times, phases,
                                suppress_errors=True)
    return pulse


def spectra_diff_fmo(basis='eig'):
    M = scipy.io.loadmat('spectra_matrices.mat')
    Hfmo = scipy.io.loadmat('Hfmo.mat')['H'].real
    # Hfmo = np.loadtxt('./Hfmo.txt')
    if basis == 'eig':
        U = scipy.linalg.eigh(Hfmo)[1]
    else:
        U = np.identity(7)
    Pfft = MetaArray(-M['M_sites_iso'].dot(np.kron(U, U)),
                     ticks=M['ticks'].reshape(-1))
    bounds = scipy.io.loadmat('pump-probe-max-min-12422.mat')
    N = np.max(bounds['max'])

    def temp(rho_ee_t):
        S = np.einsum('fi,ti->tf', Pfft, rho_ee_t)
        F = max(np.max(S.real - bounds['max'].reshape(1, -1)),
                np.max(bounds['min'].reshape(1, -1) - S.real)) / N
        return F

    return temp


def thermal_diff_fmo(basis='eig', T=53.55):
    Hfmo = scipy.io.loadmat('Hfmo.mat')['H'].real
    if basis == 'eig':
        U = scipy.linalg.eigh(Hfmo)[1]
    else:
        U = np.identity(7)
    rho_th = U.T.dot(thermal_state(Hfmo, T)).dot(U)
    return lambda rho_t: trace_distance(den(rho_t[-1]), rho_th)


def quadratic_penalty(x, x0):
    return (x > x0) * (x - x0) ** 2


def mean_dicts(dicts):
    d_mean = None
    for d in dicts:
        if d_mean is None:
            d_mean = d
        else:
            for k, v in d.iteritems():
                d_mean[k] += v
    for k in d_mean:
        d_mean[k] /= len(dicts)
    return d_mean


def averaged_error_func(func):
    def wrapper(args, return_dict=False):
        if not return_dict:
            return mean(func(args) for args in args_iter)
        else:
            return mean_dicts([func(args, return_dict=True)
                               for args in args_iter])
    return wrapper


def error_func_target_state(hamiltonian, t_extra, target_element=None,
                            target_func=None, max_abs=False,
                            pulse_maker=pair_of_gaussians,
                            anti_target=False, weak_field=True,
                            orientation='iso', opt_normalize=True,
                            t_opt_offset=0, cmp_thermal=False, duration=None,
                            basis='exciton', ensemble_size=1, **simulate_opts):

    def error_func(args, return_dict=False):
        if orientation == 'opt':
            x, y = args[:2]
            orient = (cos(x) * cos(y), cos(x) * sin(y), -sin(x))
            polar = (orient, orient)
            args = args[2:]

        if duration == 'opt':
            pulse = pulse_maker(args[1:], duration=args[0])
        else:
            pulse = pulse_maker(args)

        error = 0

        error += quadratic_penalty(pulse.t_final - pulse.t_init, 5000)

        args = (hamiltonian, t_extra)
        kargs = dict(pump=pulse, **simulate_opts)
        if orientation == 'opt':
            rho = simulate_pump(*args, polarization=polar, **kargs)
        elif orientation == 'iso':
            if not weak_field:
                raise NotImplementedError('only weak field is implemented')
            rho = isotropic_average_2nd_order(simulate_pump)(*args, **kargs)

        rho_after = rho[rho.ticks > pulse.t_final + t_opt_offset]
        if opt_normalize:
            rho_opt = normalized_exclude_ground(rho_after)
        else:
            rho_opt = meta_series_op(exclude_ground)(rho_after)

        if cmp_thermal:
            rho_therm = den_to_vec(hamiltonian.thermal_state).reshape(1, -1)
            rho_opt = np.append(rho_opt, rho_therm, axis=0)

        rho_ret = rho_opt.copy()
        if basis == 'exciton':
            trans = hamiltonian.system.ref_system.site_to_exc
            rho_opt = np.array([trans(r, 'e') for r in rho_opt])
        elif basis != 'site':
            raise NotImplementedError('invalid basis')

        if target_element is not None:
            if max_abs:
                F = np.abs(rho_opt[:, target_element])
            else:
                F = rho_opt[:, target_element].real
        else:
            F = target_func(rho_opt)

        error_vec = error + (F if anti_target else 1 - F)

        if weak_field:
            pop = 1 - rho_after[:, 0].real
            error_vec[:len(pop)] += 10e3 * quadratic_penalty(pop, 0.01)

        error += float(np.min(error_vec))

        if return_dict:
            ticks_opt = rho.ticks[rho.ticks > pulse.t_final + t_opt_offset]
            if cmp_thermal:
                ticks_opt = np.append(ticks_opt, [np.infty])
            return dict(rho_opt=rho_ret, error=error, error_vec=error_vec,
                        ticks_opt=ticks_opt,
                        ground_pop=float(rho_after[-1, 0].real),
                        t_end=ticks_opt[np.argmin(error_vec)])
        else:
            return error
    return error_func


def bound_func(f):
    def temp(args):
        args_scaled = np.max(np.c_[np.min(np.c_[np.array(args).reshape(-1, 1),
                                                np.ones((len(args), 1))],
                                   1).reshape(-1, 1),
                                   np.zeros((len(args), 1))], 1)
        return f(args_scaled)
    return temp


def fidelity(rho, sigma):
    sqrtm = scipy.linalg.sqrtm
    return np.trace(sqrtm(sqrtm(rho).dot(sigma).dot(sqrtm(rho)))).real


def trace_distance(rho, sigma):
    sqrtm = scipy.linalg.sqrtm
    A = rho - sigma
    return np.trace(sqrtm(A.conj().T.dot(A))).real / 2.


def res_print(func, print_args=False):
    res_cache = set()

    def temp(*args):
        res = func(*args)
        if res not in res_cache:
            res_cache.add(res)
            if print_args:
                print args, ' -> ', res
            else:
                print res
    return temp
