"""
Cluster mass fitting using the P(z) method of Applegate et al. (2014). Henceforth "WtG"
for "Weighing the Giants".
"""

from typing import cast, Sequence, Tuple

import numpy as np
import pandas as pd
from numcosmo_py import Ncm, Nc
from numpy.typing import NDArray

from ..utils.utils import compute_tangential_component, CoordSystem
from .mset import rebuild_mset, extract_mset_params
from .likelihood import Likelihood

import emcee
from multiprocessing import get_context
from pydantic import Field, ConfigDict, PrivateAttr, field_validator
from scipy.optimize import minimize


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


def get_data_vector_wtg(
    data: pd.DataFrame,
    radius_bounds: Tuple[float, float],
    mset: Ncm.MSet,
    coord_system: CoordSystem,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Sequence[Ncm.Spline],
]:
    """
    Calculate the calibrated shear estimate for each galaxy in the dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset containing ellipticity measurements, noise, and bias information.
    mset : Ncm.MSet
        The model set.

    Returns
    -------
    Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        Sequence[Ncm.Spline],
    ]
        A tuple containing the calibrated shear estimates, radii, and photometric redshift splines for each galaxy.
    """
    data_cut = data.copy()
    r_min, r_max = radius_bounds
    data_cut = data_cut[
        (data_cut["radius"] >= r_min) & (data_cut["radius"] <= r_max)
    ].reset_index(drop=True)
    # Compute shear estimate
    e1 = np.array(data_cut["i_hsmshaperegauss_e1"].values.astype(np.float64))
    e2 = np.array(data_cut["i_hsmshaperegauss_e2"].values.astype(np.float64))
    ra = np.array(data_cut["i_ra"].values.astype(np.float64))
    dec = np.array(data_cut["i_dec"].values.astype(np.float64))
    e_noise = np.array(
        data_cut["i_hsmshaperegauss_derived_sigma_e"].values.astype(np.float64)
    )
    e_rms = np.array(
        data_cut["i_hsmshaperegauss_derived_rms_e"].values.astype(np.float64)
    )
    mult_bias = np.array(
        data_cut["i_hsmshaperegauss_derived_shear_bias_m"].values.astype(np.float64)
    )
    additive_bias1 = np.array(
        data_cut["i_hsmshaperegauss_derived_shear_bias_c1"].values.astype(np.float64)
    )
    additive_bias2 = np.array(
        data_cut["i_hsmshaperegauss_derived_shear_bias_c2"].values.astype(np.float64)
    )

    weights = compute_inverse_variance_weight(e_noise, e_rms)
    responsivity = compute_responsivity(e_rms, weights)
    mean_mult_bias = compute_mean_multiplicative_bias(mult_bias, weights)
    g1 = compute_shear_estimate(e1, additive_bias1, mean_mult_bias, responsivity)
    g2 = compute_shear_estimate(e2, additive_bias2, mean_mult_bias, responsivity)

    # Rescale noise to shear-estimate scale to match the g = e/(2R(1+m)) calibration
    calib = 2.0 * responsivity * (1.0 + mean_mult_bias)
    e_rms = e_rms / calib
    e_noise = e_noise / calib

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

    gt = np.array(gt).astype(np.float64)
    radius = np.array(data_cut["radius"].values.astype(np.float64))
    pz = data_cut["pz_spline"].values.astype(Ncm.Spline).tolist()

    return gt, e_rms, e_noise, radius, pz


def precompute_wtg(
    data: pd.DataFrame,
    radius_bounds: Tuple[float, float],
    mset: Ncm.MSet,
    coord_system: CoordSystem,
    n_z: int = 45,
) -> dict:
    """
    Pre-compute everything that does not depend on theta for the vectorized
    wtg likelihood. Each galaxy gets its own z-grid spanning its P(z)
    spline bounds with composite Simpson weights, so the spline is never
    evaluated outside its support.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset.
    mset : Ncm.MSet
        The model set.
    n_z : int
        Number of z-nodes per galaxy. Must be odd (Simpson's rule).

    Returns
    -------
    dict
        Cached arrays and flattened (R, z) vectors consumed by
        ``ln_likelihood_wtg_vec``.
    """
    assert n_z % 2 == 1, "n_z must be odd (Simpson's rule)"

    gt, e_rms, e_noise, radius, pz_splines = get_data_vector_wtg(
        data, radius_bounds, mset, coord_system
    )
    n_gal = len(gt)

    z_per_gal = np.empty((n_gal, n_z))
    pz_per_gal = np.empty((n_gal, n_z))
    w_per_gal = np.empty((n_gal, n_z))

    base_w = np.ones(n_z)
    base_w[1:-1:2] = 4.0
    base_w[2:-2:2] = 2.0

    for i, sp in enumerate(pz_splines):
        zi, zf = sp.get_bounds()
        zg = np.linspace(zi, zf, n_z)
        z_per_gal[i] = zg
        pz_per_gal[i] = [sp.eval(float(z)) for z in zg]
        h = (zf - zi) / (n_z - 1)
        w_per_gal[i] = base_w * (h / 3.0)

    sigma2 = e_rms**2 + e_noise**2
    log_norm = -0.5 * np.log(sigma2) - 0.5 * np.log(2.0 * np.pi)

    return {
        "gt": gt,
        "sigma2": sigma2,
        "log_norm": log_norm,
        "n_gal": n_gal,
        "n_z": n_z,
        "R_flat": np.repeat(radius, n_z).tolist(),
        "z_flat": z_per_gal.ravel().tolist(),
        "pz_w": pz_per_gal * w_per_gal,
    }


def ln_likelihood_wtg(
    theta: NDArray[np.float64],
    pre: dict,
    mset: Ncm.MSet,
) -> float:
    """
    Vectorized log-likelihood for the wtg method.

    Replaces the per-galaxy ``scipy.integrate.quad`` loop with a single
    batched call to ``reduced_shear_array_equal`` over flattened
    ``(N_gal * n_z)`` (R, z) pairs, plus an inline Gaussian.

    Parameters
    ----------
    theta : NDArray[np.float64]
        The model parameters (currently just log10MDelta).
    pre : dict
        Output of ``precompute_wtg``.
    mset : Ncm.MSet
        The model set.
    fit_sigma : bool
        Whether to include the sigma parameters in the likelihood.

    Returns
    -------
    float
        The log-likelihood value (including prior).
    """
    fparams = pre["fparams"]
    fparams_bounds = pre["fparams_bounds"]

    if fparams_bounds is not None:
        for value, (lower, upper) in zip(theta, fparams_bounds):
            if not (lower < value < upper):
                return -np.inf

    for value, param in zip(theta, fparams):
        model_name, param_name = param.split(":", 1)
        mset.peek_by_name(model_name)[param_name] = float(value)

    smd = cast(Nc.WLSurfaceMassDensity, mset.peek_by_name("NcWLSurfaceMassDensity"))
    dp = cast(Nc.HaloDensityProfile, mset.peek_by_name("NcHaloDensityProfile"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))

    z_l = float(hp["z"])

    flat = smd.reduced_shear_array_equal(
        dp,
        cosmo,
        pre["R_flat"],
        1.0,
        1.0,
        pre["z_flat"],
        z_l,
        z_l,
    )
    gt_theory = np.asarray(flat).reshape(pre["n_gal"], pre["n_z"])

    gt = pre["gt"][:, None]
    sigma2 = pre["sigma2"][:, None]
    log_norm = pre["log_norm"][:, None]
    log_g = -0.5 * (gt - gt_theory) ** 2 / sigma2 + log_norm

    integrals = np.einsum("ij,ij->i", np.exp(log_g), pre["pz_w"])
    integrals = np.where(integrals > 0, integrals, np.finfo(float).tiny)

    return float(np.sum(np.log(integrals)))


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
    return ln_likelihood_wtg(
        theta,
        _STATE["pre"],
        _STATE["mset"],
    )


class WtGLikelihood(Likelihood):
    """
    Likelihood class for the binned excess surface mass density profile described in
    Umetsu et al. (2020). Uses shapeHSM products.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    radius_bounds: Tuple[float, float] = Field(
        default=(0.4, 4.0),
        description=(
            "Minimum and maximum radii (in Mpc) to include in the likelihood."
        ),
    )

    _gt: NDArray[np.float64] = PrivateAttr()
    _e_rms: NDArray[np.float64] = PrivateAttr()
    _e_noise: NDArray[np.float64] = PrivateAttr()
    _radius: NDArray[np.float64] = PrivateAttr()
    _pz: Sequence[Ncm.Spline] = PrivateAttr()
    _pre: dict = PrivateAttr()

    @field_validator("radius_bounds")
    def validate_radius_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if any(r < 0 for r in v):
            raise ValueError("radius_bounds must be non-negative")
        if v[0] >= v[1]:
            raise ValueError("radius_bounds must be (min, max) with min < max")
        return v

    def prepare_data(self) -> None:
        """Compute the data vector and related quantities from the observation data."""
        (
            self._gt,
            self._e_rms,
            self._e_noise,
            self._radius,
            self._pz,
        ) = get_data_vector_wtg(
            self.obs, self.radius_bounds, self.mset, self.coord_system
        )
        self._pre = precompute_wtg(
            self.obs, self.radius_bounds, self.mset, self.coord_system, 45
        )

    def maximum_likelihood_estimate(self) -> NDArray[np.float64]:
        """Return best-fit [log10M] via scipy.optimize.minimize."""
        self._pre["fparams"] = list(self.fparams)
        self._pre["fparams_bounds"] = (
            list(self.fparams_bounds) if self.fparams_bounds is not None else None
        )

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

        result = minimize(
            lambda theta: -ln_likelihood_wtg(
                theta,
                self._pre,
                self.mset,
            ),
            x0=initial_values,
            method="Nelder-Mead",
        )
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
        filename: str = "wtg_mcmc.h5",
    ) -> pd.DataFrame:
        """Run emcee with per-process mset reconstruction."""
        self._pre["fparams"] = list(self.fparams)
        self._pre["fparams_bounds"] = (
            list(self.fparams_bounds) if self.fparams_bounds is not None else None
        )
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
        initial_values = np.tile(np.asarray(default_values), (nwalkers, 1))
        initial_values += (
            0.1
            * np.random.randn(nwalkers, len(self.fparams))
            * np.asarray(default_scale)
        )
        backend = emcee.backends.HDFBackend(filename)

        if nthreads <= 1:
            # Sequential execution for ultra-fast likelihoods
            init_worker(self._pre, params)
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
            # Use a fork context so workers inherit module/global state instead
            # of re-importing (Python 3.14 defaults to forkserver, which would
            # re-run the calling module / break inside notebooks and nested procs).
            with get_context("fork").Pool(
                nthreads, initializer=init_worker, initargs=(self._pre, params)
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
