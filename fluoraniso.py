from __future__ import division
from typing import Union
import numpy as np
from functools import partial
from scipy import optimize


def simple_binding_function(total_binder: Union[np.ndarray, float], kd: float, aniso_bound: float, aniso_free: float,
                            ligand: float):
    """
    Calculates the predicted amount of binding for a non-cooperative interaction, taking into account receptor
    depletion.

    Parameters
    ----------
    total_binder : float or ndarray
        The total (bound + unbound) concentration of the unlabeled protein/receptor molecule
    kd : float
        The dissociation constant for the interaction
    aniso_bound : float
        The signal given off by a bound labeled ligand
    aniso_free : float
        The signal given off by an unbound/free ligand
    ligand : float
        The total (bound + unbound) concentration of labeled ligand present

    Returns
    -------
    A value (or array) scaled to the interval [`aniso_free`:`aniso_bound`] representing the fraction of bound ligand.

    Notes
    -----
    `total_binder`, `kd`, and `ligand` should all be in the same concentration units (e.g., nanomolar)
    """
    return aniso_free + (aniso_bound - aniso_free) * ((ligand + kd + total_binder)
                                                      - np.sqrt((np.power(-ligand - kd - total_binder, 2)
                                                                 - 4 * ligand * total_binder))) / (2 * ligand)


def hill_function(free_binder: Union[np.ndarray, float], kd: float, aniso_bound: float, aniso_free: float, n: float):
    """
    Calculates the predicted amount of binding for a cooperative binding interaction (as defined by the Hill equation).
    Note that there is no way to explicitly take receptor depletion into account in this case.

    Parameters
    ----------
    free_binder : float or ndarray
        The concentration of free/unbound protein/receptor molecule
    kd : float
        The dissociation constant for the interaction
    aniso_bound : float
        The signal given off by a bound labeled ligand
    aniso_free : float
        The signal given off by an unbound/free ligand
    n : float
        The Hill coefficient for the interaction

    Returns
    -------
    A value (or array) scaled to the interval [`aniso_free`:`aniso_bound`] representing the fraction of bound ligand.

    Notes
    -----
    `free_binder`, `kd`, and `ligand` should all be in the same concentration units (e.g., nanomolar)
    """
    with np.errstate(over='ignore', under='ignore', divide='ignore'):
        return aniso_free + (aniso_bound - aniso_free) / (1.0 + np.power(kd / free_binder, n))


def do_noncooperative_fit(total_binder: np.ndarray, anisotropies: np.ndarray, std: np.ndarray, ligand: float):
    """
    Given measurements of bound labeled ligand (on an arbitrary scale) as a function of total receptor
    concentration (as the ligand concentration is held constant), fit a dissociation equilibrium constant.

    Parameters
    ----------
    total_binder : ndarray
        The total (bound + unbound) concentrations of the unlabeled protein/receptor molecule used in the experiment
    anisotropies :  ndarray
        The observed arbitrary values that are linearly proportional to bound ligand
    std : ndarray
        The uncertainties (standard error) of the observations in `anisotropies`
    ligand : float
        The constant ligand concentration present across the experiment

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters to `simple_binding_function` (Kd, the arbitrary value for bound ligand, and
        the arbitrary value for the unbound ligand)
    pcov : 2d ndarray
        The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).

    Notes
    -----
    The return values are simply the results of a call to `scipy.optimize.curve_fit`

    `total_binder` and `ligand` should all be in the same concentration units (e.g., nanomolar). This will be the unit
    of the results as well.

    This function assumes for sanity that Kd and the measurement scale used to represent binding cannot be negative.
    This assertion is represented by asserting (0.0, np.inf) as bounds for all fitted variables in the call to
     `scipy.optimize.curve_fit`
    """
    # Curry the binding function to avoid fitting the ligand concentration
    func = partial(simple_binding_function, ligand=ligand)
    return optimize.curve_fit(func, total_binder, anisotropies, sigma=std, absolute_sigma=True, bounds=(0.0, np.inf))


def do_hill_fit(total_binder: np.ndarray, anisotropies: np.ndarray, std: np.ndarray, ligand: float):
    """
    Given measurements of bound labeled ligand (on an arbitrary scale) as a function of total receptor
    concentration (as the ligand concentration is held constant), fit a dissociation equilibrium constant and a Hill
    coefficient to the data.

    Things get tricky here because the Hill equation cannot be explicitly modified to account for the phenomenon of
    receptor depletion, yet experimental data is recorded as a function of the total amount of receptor present. This
    function gets around this by running two fits: One just to estimate the bound fraction, then one to accurately fit
    the Kd and Hill coefficient.

    Parameters
    ----------
    total_binder : ndarray
        The total (bound + unbound) concentrations of the unlabeled protein/receptor molecule used in the experiment
    anisotropies :  ndarray
        The observed arbitrary values that are linearly proportional to bound ligand
    std : ndarray
        The uncertainties (standard error) of the observations in `anisotropies`
    ligand : float
        The constant ligand concentration present across the experiment

    Returns
    -------
    popt : ndarray
        Optimal values for the parameters to `hill_function` (Kd, the arbitrary value for bound ligand, the arbitrary
        value for the unbound ligand, and the Hill coefficient)
    pcov : 2d ndarray
        The estimated covariance of popt. The diagonals provide the variance of the parameter estimate.
        To compute one standard deviation errors on the parameters use perr = np.sqrt(np.diag(pcov)).

    Notes
    -----
    The return values are simply the results of a call to `scipy.optimize.curve_fit`

    `total_binder` and `ligand` should all be in the same concentration units (e.g., nanomolar). This will be the unit
    of the results as well.

    This function assumes for sanity that Kd and the measurement scale used to represent binding cannot be negative.
    This assertion is represented by asserting (0.0, np.inf) as bounds for all fitted variables in the call to
     `scipy.optimize.curve_fit`. The Hill coefficient, however, can be negative.
    """
    # bounds: Hill coefficients can be negative, unlike the other params
    bounds = ([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    # first do a preliminary fit
    popt, pcov = optimize.curve_fit(hill_function, total_binder, anisotropies, sigma=std, absolute_sigma=True,
                                    bounds=bounds)
    # ... the Kd reported by this naive fit does not account for receptor depletion, but it will give estimates of
    # Afree and Abound from which we can estimate it
    abound, afree = popt[1], popt[2]
    depletion = (anisotropies - afree) / (abound - afree) * ligand
    # bound the amount of depletion to between Nothing and the total amount of ligand
    # also bound the amount of free binder to be >= 0.0
    depletion[depletion < 0.0] = 0.0
    depletion[depletion > ligand] = ligand
    adjusted_binder = total_binder - depletion
    adjusted_binder[adjusted_binder < 0.0] = 0.0

    return optimize.curve_fit(hill_function, adjusted_binder, anisotropies, sigma=std, absolute_sigma=True,
                              bounds=bounds)


def simulate_cooperative_binding(total_binder: np.ndarray, kd: float, aniso_bound: float, aniso_free: float, n: float,
                                 ligand: float):
    """
    Generate a simulated cooperative binding curve using total (unbound + bound) receptor concentrations as a starting
    point.

    The results will be scaled to an arbitrary range (`aniso_free`, `aniso_bound`).

    Uses the same bootstrapping strategy as `do_hill_fit`. This function should only be used to plot theoretical values,
    not for fitting.

    Parameters
    ----------
    total_binder : ndarray
        The concentration of free/unbound protein/receptor molecule
    kd : float
        The dissociation constant for the interaction
    aniso_bound : float
        The signal given off by a bound labeled ligand
    aniso_free : float
        The signal given off by an unbound/free ligand
    n : float
        The Hill coefficient for the interaction
    ligand : float
        The constant ligand concentration to account for

    Returns
    -------
    An array scaled to the interval [`aniso_free`:`aniso_bound`] representing the fraction of bound ligand.
    """
    # Similar to the bootstrapping method to take receptor depletion into account for fitting
    raw_aniso = hill_function(total_binder, kd, aniso_bound, aniso_free, n)
    depletion = (raw_aniso - aniso_free) / (aniso_bound - aniso_free) * ligand
    return hill_function(total_binder - depletion, kd, aniso_bound, aniso_free, n)
