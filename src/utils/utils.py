"""
Utility functions for mass fitting.
"""

from typing import cast
import numpy as np
from numcosmo_py import Ncm, Nc
from numpy.typing import NDArray


def create_ncm_spline(
    pz: NDArray[np.float64], nodes: NDArray[np.float64]
) -> Ncm.Spline | None:
    """
    Create a NumCosmo Spline from the given P(z) and nodes.

    Parameters
    ----------
    pz : NDArray[np.float64]
        The P(z) values.
    nodes : NDArray[np.float64]
        The nodes corresponding to the P(z) values.

    Returns
    -------
    Ncm.Spline
        The created NumCosmo Spline.
    """
    lower_index = 0
    upper_index = len(nodes) - 1

    for j in range(len(nodes)):
        if pz[j] / max(pz) > 1e-3:
            lower_index = j
            break

    for j in range(len(nodes) - 1, -1, -1):
        if pz[j] / max(pz) > 1e-3:
            upper_index = j
            break

    if upper_index - lower_index < 6:
        max_up_delta = min(
            len(nodes) - 1 - upper_index, 6 - (upper_index - lower_index)
        )
        upper_index += max_up_delta
        lower_index -= (upper_index - lower_index) - max_up_delta

    xv = Ncm.Vector.new(upper_index - lower_index + 1)
    yv = Ncm.Vector.new(upper_index - lower_index + 1)

    for j in range(0, upper_index - lower_index + 1):
        xv.set(j, nodes[lower_index + j])
        yv.set(j, pz[lower_index + j])

    pz_spline = Ncm.SplineCubicNotaknot.new_full(xv, yv, True)

    pz_spline.prepare()

    norm = pz_spline.eval_integ(xv.get(0), xv.get(xv.len() - 1))

    for j in range(0, yv.len()):
        yv.set(j, yv.get(j) / norm)

    pz_spline.prepare()

    return pz_spline


def compute_radius(ra: float, dec: float, mset: Ncm.MSet) -> float:
    """
    Compute the radius from the given right ascension and declination.

    Parameters
    ----------
    ra : float
        The right ascension of the object.
    dec : float
        The declination of the object.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.

    Returns
    -------
    float
        The computed radius in Mpc.
    """
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))

    hp.prepare(cosmo)

    return hp.projected_radius_from_ra_dec(cosmo, ra, dec)


def compute_tangential_component(
    e1: float, e2: float, ra: float, dec: float, mset: Ncm.MSet
) -> float:
    """
    Compute the tangential component of the ellipticity from the given shape parameters
    and position.

    Parameters
    ----------
    e1 : float
        The first component of the ellipticity.
    e2 : float
        The second component of the ellipticity.
    ra : float
        The right ascension of the object.
    dec : float
        The declination of the object.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.

    Returns
    -------
    float
        The computed tangential component of the ellipticity.
    """
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))

    hp.prepare(cosmo)

    _, phi = hp.polar_angles(ra, dec)
    phi = np.pi - phi

    return np.real((e1 + 1j * e2) * np.exp(-2j * phi))


def compute_cross_component(
    e1: float, e2: float, ra: float, dec: float, mset: Ncm.MSet
) -> float:
    """
    Compute the cross component of the ellipticity from the given shape parameters and
    position.

    Parameters
    ----------
    e1 : float
        The first component of the ellipticity.
    e2 : float
        The second component of the ellipticity.
    ra : float
        The right ascension of the object.
    dec : float
        The declination of the object.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.

    Returns
    -------
    float
        The computed cross component of the ellipticity.
    """
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))

    hp.prepare(cosmo)

    _, phi = hp.polar_angles(ra, dec)
    phi = np.pi - phi

    return np.imag((e1 + 1j * e2) * np.exp(-2j * phi))
