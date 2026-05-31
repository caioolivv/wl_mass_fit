"""
Cluster mass fit using the Umtesu et al. (2020) model.
"""

from .likelihood import Likelihood
from .mset import extract_mset_params, rebuild_mset
from ..utils.utils import compute_tangential_component, CoordSystem
from typing import cast, Sequence, Tuple
from pydantic import ConfigDict, Field, field_validator, PrivateAttr

import numpy as np
import pandas as pd
from scipy.integrate import quad
from numcosmo_py import Ncm, Nc
from numpy.typing import NDArray
from scipy.optimize import minimize
import emcee
from multiprocessing import Pool, get_context


def compute_inverse_variance_weight(
    e_noise: NDArray[np.float64], e_rms: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the inverse variance weight.

    Parameters
    ----------
    e_noise : NDArray[np.float64]
        The noise in the ellipticity measurement.
    e_rms : NDArray[np.float64]
        The root mean square of the intrinsic ellipticity distribution.

    Returns
    -------
    NDArray[np.float64]
        The inverse variance weight.
    """
    return 1.0 / (e_noise**2 + e_rms**2)


def compute_responsivity(
    e_rms: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """
    Calculate the responsivity.

    Parameters
    ----------
    e_rms : NDArray[np.float64]
        The root mean square of the intrinsic ellipticity distribution.
    weights : NDArray[np.float64]
        The inverse variance weights for each galaxy.

    Returns
    -------
    float
        The responsivity.
    """
    return 1.0 - np.sum(weights * e_rms**2) / np.sum(weights)


def compute_mean_multiplicative_bias(
    mult_bias: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    """
    Calculate the mean multiplicative bias.

    Parameters
    ----------
    mult_bias : NDArray[np.float64]
        The multiplicative bias for each galaxy.
    weights : NDArray[np.float64]
        The inverse variance weights for each galaxy.

    Returns
    -------
    float
        The mean multiplicative bias.
    """
    return np.sum(weights * mult_bias) / np.sum(weights)


def compute_shear_estimate(
    e: NDArray[np.float64],
    additive_bias: NDArray[np.float64],
    mean_mult_bias: float,
    responsivity: float,
) -> NDArray[np.float64]:
    """
    Calculate the shear estimate.

    Parameters
    ----------
    e : NDArray[np.float64]
        The ellipticity measurements for each galaxy.
    additive_bias : NDArray[np.float64]
        The additive bias for each galaxy.
    mean_mult_bias : float
        The mean multiplicative bias.
    responsivity : float
        The responsivity.

    Returns
    -------
    NDArray[np.float64]
        The shear estimate for each galaxy.
    """
    return (e / (2 * responsivity) - additive_bias) / (1 + mean_mult_bias)


def get_data_vector_binned_shear(
    bin_edges: Sequence[float],
    data: pd.DataFrame,
    mset: Ncm.MSet,
    coord_system: CoordSystem,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """

    Parameters
    ----------
    bin_edges : Sequence[float]
        The edges of the radial bins for the calculation.
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.
    coord_system : CoordSystem
        The coordinate system to use for the tangential and cross component calculations.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        A tuple containing the mean tangential shear, mean radius, mean redshift, and
        inverse covariance matrix for the binned data.
    """
    e1 = np.array(data["i_hsmshaperegauss_e1"].values.astype(np.float64))
    e2 = np.array(data["i_hsmshaperegauss_e2"].values.astype(np.float64))
    ra = np.array(data["i_ra"].values.astype(np.float64))
    dec = np.array(data["i_dec"].values.astype(np.float64))
    e_noise = np.array(
        data["i_hsmshaperegauss_derived_sigma_e"].values.astype(np.float64)
    )
    e_rms = np.array(data["i_hsmshaperegauss_derived_rms_e"].values.astype(np.float64))
    mult_bias = np.array(
        data["i_hsmshaperegauss_derived_shear_bias_m"].values.astype(np.float64)
    )
    additive_bias1 = np.array(
        data["i_hsmshaperegauss_derived_shear_bias_c1"].values.astype(np.float64)
    )
    additive_bias2 = np.array(
        data["i_hsmshaperegauss_derived_shear_bias_c2"].values.astype(np.float64)
    )

    weights = compute_inverse_variance_weight(e_noise, e_rms)
    responsivity = compute_responsivity(e_rms, weights)
    mean_mult_bias = compute_mean_multiplicative_bias(mult_bias, weights)
    g1 = compute_shear_estimate(e1, additive_bias1, mean_mult_bias, responsivity)
    g2 = compute_shear_estimate(e2, additive_bias2, mean_mult_bias, responsivity)

    # Rescale noise to the shear-estimate scale so var_mean matches gt_mean.
    calib = 2.0 * responsivity * (1.0 + mean_mult_bias)
    e_rms = e_rms / calib
    e_noise = e_noise / calib
    weights = compute_inverse_variance_weight(e_noise, e_rms)

    gt = []

    for i in range(len(e1)):
        gt.append(
            compute_tangential_component(
                g1[i],
                g2[i],
                ra[i],
                dec[i],
                coord_system,
                mset,
            )
        )

    binned_data = data.copy()
    binned_data["g_t"] = gt
    binned_data["bin"] = pd.cut(
        binned_data["radius"], bins=bin_edges, labels=False, include_lowest=True
    )

    gt_mean = binned_data.groupby("bin", observed=True)["g_t"].mean().values
    radius_mean = binned_data.groupby("bin", observed=True)["radius"].mean().values
    var_mean = (
        binned_data.groupby("bin", observed=True)
        .apply(lambda x: 1 / np.sum(weights[x.index]), include_groups=False)
        .values
    )
    z_mean = binned_data.groupby("bin", observed=True)["z"].mean().values

    valid = np.isfinite(var_mean) & np.isfinite(gt_mean)
    inverse_covariance_matrix = np.diag(1.0 / var_mean[valid])

    return (
        gt_mean[valid],
        radius_mean[valid],
        z_mean[valid],
        inverse_covariance_matrix,
    )


def get_theory_vector_binned_shear(
    radius: NDArray[np.float64],
    zs: NDArray[np.float64],
    mset: Ncm.MSet,
) -> NDArray[np.float64]:
    """ """
    smd = cast(Nc.WLSurfaceMassDensity, mset.peek_by_name("NcWLSurfaceMassDensity"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    dp = cast(Nc.HaloDensityProfile, mset.peek_by_name("NcHaloDensityProfile"))

    z_cl = hp["z"]
    gt_nfw = np.array(
        smd.reduced_shear_array_equal(dp, cosmo, radius, 1.0, 1.0, zs, z_cl, z_cl)
    )

    return gt_nfw


def ln_likelihood_binned_shear(
    theta: NDArray[np.float64],
    fparams: Sequence[str],
    fparams_bounds: Sequence[Tuple[float, float]] | None,
    radius: NDArray[np.float64],
    redshift: NDArray[np.float64],
    data_vector: NDArray[np.float64],
    inv_cov: NDArray[np.float64],
    mset: Ncm.MSet,
) -> float:
    """
    Compute the log-probability of the model given the data and the parameters.

    Parameters
    ----------
    theta : NDArray[np.float64]
        Values for the free parameters, in the same order as ``fparams``.
    fparams : Sequence[str]
        Full names of the free parameters in the form ``"ModelName:param_name"``.
    fparams_bounds : Sequence[Tuple[float, float]] | None
        Optional uniform-prior bounds per parameter, aligned with ``fparams``.
        ``None`` disables bound checking.
    radius : NDArray[np.float64]
        The radius of the data points.
    bin_mean_inv_sigma_crit : NDArray[np.float64]
        Per-bin weight-averaged
        $\\langle \\langle \\Sigma^{-1}_{\\mathrm{cr}} \\rangle \\rangle$
        for the reduced-shear correction.
    data_vector : NDArray[np.float64]
        The data vector.
    inv_cov : NDArray[np.float64]
        The inverse of the covariance matrix.
    mset : Ncm.MSet
        The model set containing the cosmology and the halo model.

    Returns
    -------
    float
        The log-probability of the model given the data and the parameters.
    """
    if fparams_bounds is not None:
        for value, (lower, upper) in zip(theta, fparams_bounds):
            if not (lower < value < upper):
                return -np.inf

    for value, param in zip(theta, fparams):
        model_name, param_name = param.split(":", 1)
        mset.peek_by_name(model_name)[param_name] = float(value)

    theory_vector = get_theory_vector_binned_shear(radius, redshift, mset)
    diff = data_vector - theory_vector

    return -0.5 * float(diff.T @ inv_cov @ diff)


# --- worker-process API (used by multiprocessing.Pool); do not call directly
# from the parent. Ncm.MSet is not picklable, so each worker rebuilds its own
# mset once at startup via init_worker and stores it in module-level state.
_STATE: dict = {}


def init_worker(pre: dict, params: dict) -> None:
    """Pool initializer: build mset once per worker and stash with pre."""
    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    _STATE["mset"] = rebuild_mset(params)
    _STATE["pre"] = pre


def _worker_log_prob(theta) -> float:
    """Per-task likelihood evaluation, reads worker-local state."""
    pre = _STATE["pre"]
    lp = ln_likelihood_binned_shear(
        theta,
        pre["fparams"],
        pre["fparams_bounds"],
        pre["radius"],
        pre["zs"],
        pre["data_vector"],
        pre["inv_cov"],
        _STATE["mset"],
    )
    return -np.inf if np.isnan(lp) else lp


def _valid_initial_positions(
    center: NDArray[np.float64],
    scale: NDArray[np.float64],
    nwalkers: int,
    pre: dict,
    params: dict,
    max_tries: int = 100,
) -> NDArray[np.float64]:
    """Return (nwalkers, ndim) initial positions that all give finite log-prob."""
    init_worker(pre, params)
    rng = np.random.default_rng()
    positions = np.empty((nwalkers, len(center)))
    filled = 0
    for _ in range(max_tries):
        candidates = center + 0.1 * rng.standard_normal((nwalkers, len(center))) * scale
        for row in candidates:
            if filled >= nwalkers:
                break
            if np.isfinite(_worker_log_prob(row)):
                positions[filled] = row
                filled += 1
        if filled >= nwalkers:
            break
    if filled == 0:
        raise RuntimeError(
            "Could not find a single valid initial walker position after "
            f"{max_tries} attempts. Check parameter bounds and data."
        )
    if filled < nwalkers:
        # Replicate valid positions found so far with tiny jitter so that all
        # walkers are valid and linearly independent (avoids emcee condition-
        # number error and the -inf - (-inf) = nan warning).
        base = positions[:filled]
        n_missing = nwalkers - filled
        indices = rng.integers(0, filled, size=n_missing)
        jitter = 1e-6 * rng.standard_normal((n_missing, len(center))) * scale
        positions[filled:] = base[indices] + jitter
    return positions


class BinnedShearLikelihood(Likelihood):
    """
    Likelihood class for the binned excess surface mass density profile described in
    Umetsu et al. (2020). Uses shapeHSM products.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    bin_edges: Sequence[float] = Field(
        default=np.logspace(np.log10(0.4), np.log10(4.0), 6),
        description="Radial bin edges in Mpc.",
    )

    _data_vector: NDArray[np.float64] = PrivateAttr()
    _radius: NDArray[np.float64] = PrivateAttr()
    _zs: NDArray[np.float64] = PrivateAttr()
    _inv_cov: NDArray[np.float64] = PrivateAttr()

    @field_validator("bin_edges")
    def validate_bin_edges(cls, v: Sequence[float]) -> Sequence[float]:
        """Ensure bin_edges are strictly increasing."""
        if any(v[i] >= v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("bin_edges must be strictly increasing")
        if any(v[i] <= 0 for i in range(len(v))):
            raise ValueError("bin_edges must be positive")
        return v

    def prepare_data(self) -> None:
        """Compute the data vector and related quantities from the observation data."""
        (
            self._data_vector,
            self._radius,
            self._zs,
            self._inv_cov,
        ) = get_data_vector_binned_shear(
            self.bin_edges, self.obs, self.mset, self.coord_system
        )

    def maximum_likelihood_estimate(self) -> NDArray[np.float64]:
        """Return best-fit [log10M] via scipy.optimize.minimize."""
        default_values = []
        default_scale = []
        for param in self.fparams:
            model_name, param_name = param.split(":", 1)
            default_values.append(self.mset.peek_by_name(model_name)[param_name])
            default_scale.append(
                self.mset.peek_by_name(model_name)
                .param_get_desc(param_name)
                .get("scale", 1.0)
            )
        initial_values = np.array(default_values)
        initial_values += 0.1 * np.random.randn(len(self.fparams)) * default_scale

        def _neg_lp(theta):
            v = -ln_likelihood_binned_shear(
                theta,
                self.fparams,
                self.fparams_bounds,
                self._radius,
                self._zs,
                self._data_vector,
                self._inv_cov,
                self.mset,
            )
            return 1e300 if not np.isfinite(v) else v

        result = minimize(_neg_lp, x0=initial_values, method="Nelder-Mead")
        # Leave mset at the optimal point, not at the last Nelder-Mead probe.
        for value, param in zip(result.x, self.fparams):
            model_name, param_name = param.split(":", 1)
            self.mset.peek_by_name(model_name)[param_name] = float(value)
        return result.x

    def sample_posterior(
        self,
        nsamples: int,
        nwalkers: int = 32,
        nthreads: int = 4,
        progress: bool = True,
        filename: str = "binned_shear_mcmc.h5",
    ) -> pd.DataFrame:
        """Run emcee with per-process mset reconstruction."""

        pre = {
            "fparams": list(self.fparams),
            "fparams_bounds": (
                list(self.fparams_bounds) if self.fparams_bounds is not None else None
            ),
            "data_vector": self._data_vector,
            "radius": self._radius,
            "zs": self._zs,
            "inv_cov": self._inv_cov,
        }
        params = extract_mset_params(self.mset)
        default_values = []
        default_scale = []
        for param in self.fparams:
            model_name, param_name = param.split(":", 1)
            default_values.append(self.mset.peek_by_name(model_name)[param_name])
            default_scale.append(
                self.mset.peek_by_name(model_name)
                .param_get_desc(param_name)
                .get("scale", 1.0)
            )
        center = np.asarray(default_values)
        scale = np.asarray(default_scale)
        initial_values = _valid_initial_positions(
            center, scale, nwalkers, pre, params
        )
        backend = emcee.backends.HDFBackend(filename)

        if nthreads <= 1:
            # Sequential execution for ultra-fast likelihoods
            init_worker(pre, params)
            sampler = emcee.EnsembleSampler(
                nwalkers,
                len(self.fparams),
                _worker_log_prob,
                backend=backend,
            )
            sampler.run_mcmc(
                initial_values,
                nsamples - cast(int, backend.iteration),
                progress=progress,
            )
        else:
            with Pool(
                nthreads, initializer=init_worker, initargs=(pre, params)
            ) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    len(self.fparams),
                    _worker_log_prob,
                    backend=backend,
                    pool=pool,
                )
                sampler.run_mcmc(
                    initial_values,
                    nsamples - cast(int, backend.iteration),
                    progress=progress,
                )

        flat = sampler.get_chain(flat=True)
        posterior = pd.DataFrame(
            flat, columns=[p.split(":", 1)[1] for p in self.fparams]
        )
        self._posterior = posterior

        return posterior
