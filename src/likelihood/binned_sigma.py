"""
Cluster mass fit using binned excess surface mass density profile described in
Umetsu et al. (2020). Uses shapeHSM products.
"""

from .likelihood import Likelihood
from .mset import extract_mset_params, rebuild_mset
from multiprocessing import get_context
from typing import cast, Sequence, Tuple
import emcee
from pydantic import ConfigDict, Field, field_validator, PrivateAttr
from numcosmo_py import Ncm, Nc
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize


def compute_mean_inv_sigma_critical(
    pz_spline: Ncm.Spline,
    mset: Ncm.MSet,
) -> float:
    """
    Calculate the P(z)-weighted mean inverse critical surface mass density
    $\\langle \\Sigma^{-1}_{\\mathrm{cr}} \\rangle$ for the galaxy (Eq. 6).

    Parameters
    ----------
    pz_spline : Ncm.Spline
        The spline representation of the P(z) distribution for the galaxy.
    mset : Ncm.MSet
        The model set containing cosmology, surface mass density, and halo position.

    Returns
    -------
    float
        The P(z)-weighted mean inverse critical surface mass density
        $\\langle \\Sigma^{-1}_{\\mathrm{cr}} \\rangle$.
    """
    pz_spline.prepare()

    smd = cast(Nc.WLSurfaceMassDensity, mset.peek_by_name("NcWLSurfaceMassDensity"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))

    z_cluster = hp["z"]
    zi, zf = pz_spline.get_bounds()
    norm = pz_spline.eval_integ(zi, zf)
    integrand = lambda z: pz_spline.eval(z) / smd.sigma_critical(
        cosmo, z, z_cluster, z_cluster
    )

    return quad(integrand, zi, zf)[0] / norm


def compute_weight_factor(
    mean_inv_sigma_crit: NDArray[np.float64],
    sigma_err: NDArray[np.float64],
    sigma_rms: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the weight factor for each galaxy (Eq. 7).

    Parameters
    ----------
    mean_inv_sigma_crit : NDArray[np.float64]
        The P(z)-weighted mean inverse critical surface mass density for each galaxy.
    sigma_err : NDArray[np.float64]
        The measurement error of the ellipticity for each galaxy.
    sigma_rms : NDArray[np.float64]
        The intrinsic shape noise (rms of ellipticity) for each galaxy.

    Returns
    -------
    NDArray[np.float64]
        The weight factor for each galaxy.
    """
    return mean_inv_sigma_crit**2 / (sigma_err**2 + sigma_rms**2)


def compute_mult_bias_correction(data: pd.DataFrame) -> float:
    """
    Calculate the multiplicative bias correction for each galaxy in the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.

    Returns
    -------
    NDArray[np.float64]
        An array of multiplicative bias corrections for each galaxy.
    """
    denominator = data["w"].sum()

    if denominator == 0:
        return np.nan

    return (
        data["w"] * (1 + data["i_hsmshaperegauss_derived_shear_bias_m"])
    ).sum() / denominator


def compute_responsivity_correction(data: pd.DataFrame) -> float:
    """
    Calculate the responsivity correction for each galaxy in the data.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.

    Returns
    -------
    float
        The responsivity correction for the galaxy sample.
    """
    denominator = data["w"].sum()

    if denominator == 0:
        return np.nan

    return (
        1
        - (data["w"] * data["i_hsmshaperegauss_derived_rms_e"] ** 2).sum() / denominator
    )


def compute_effective_radius(data: pd.DataFrame) -> float:
    """
    Calculate the effective radius for the galaxy sample.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.

    Returns
    -------
    float
        The effective radius for the galaxy sample.
    """
    denominator = (data["w"] / data["radius"]).sum()

    if denominator == 0:
        return np.nan

    return data["w"].sum() / denominator


def compute_delta_sigma(
    responsivity: float, mult_correction: float, data: pd.DataFrame
) -> float:
    """
    Calculate the average excess surface mass density (Delta Sigma) for the galaxy
    sample.

    Parameters
    ----------
    responsivity : float
        The responsivity correction for the galaxy sample.
    mult_correction : float
        The multiplicative bias correction for the galaxy sample.
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.

    Returns
    -------
    float
        The average excess surface mass density for the galaxy sample.
    """
    denominator = data["w"].sum() * mult_correction * 2 * responsivity

    if denominator == 0:
        return np.nan

    return (data["w"] * data["e_t"] / data["mean_inv_sigma_crit"]).sum() / denominator


def compute_var_shape(
    responsivity: float, mult_correction: float, data: pd.DataFrame
) -> float:
    """
    Calculate the variance of the shape noise for the galaxy sample.

    Parameters
    ----------
    responsivity : float
        The responsivity correction for the galaxy sample.
    mult_correction : float
        The multiplicative bias correction for the galaxy sample.
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.

    Returns
    -------
    float
        The variance of the shape noise for the galaxy sample.
    """
    return 1 / (4 * responsivity**2 * mult_correction**2 * data["w"].sum())


def get_data_vector_binned_sigma(
    bin_edges: Sequence[float], data: pd.DataFrame, mset: Ncm.MSet
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """
    Compute the data vector for the galaxy sample using the binned_sigma et al. (2020) model.

    Parameters
    ----------
    bin_edges : Sequence[float]
        The edges of the radial bins for the calculation.
    data : pd.DataFrame
        The data containing the necessary columns for the calculation.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.

    Returns
    -------
    tuple[NDArray, NDArray, NDArray, NDArray]
        A tuple containing (delta_sigma_values, effective_radius_values,
        bin_mean_inv_sigma_crit, inverse_covariance_matrix).
        bin_mean_inv_sigma_crit is the per-bin weight-averaged
        $\\langle \\langle \\Sigma^{-1}_{\\mathrm{cr}} \\rangle \\rangle$ needed
        for the reduced-shear correction (Eq. 22).
    """
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    z_cluster = float(hp["z"])

    binned_data = data.copy()
    binned_data["bin"] = pd.cut(
        binned_data["radius"], bins=bin_edges, labels=False, include_lowest=True
    )

    effective_radius_values = []
    delta_sigma_values = []
    bin_mean_inv_sigma_crit_values = []
    var_shape_values = []

    for bin_idx in range(len(bin_edges) - 1):
        bin_data = binned_data[binned_data["bin"] == bin_idx].copy()

        if bin_data.empty:
            effective_radius_values.append(np.nan)
            delta_sigma_values.append(np.nan)
            var_shape_values.append(np.nan)
            bin_mean_inv_sigma_crit_values.append(np.nan)
            continue

        bin_data["mean_inv_sigma_crit"] = bin_data.apply(
            lambda row: _mean_inv_sigma_crit_cached(row["pz_spline"], mset, z_cluster),
            axis=1,
        )
        bin_data["w"] = bin_data.apply(
            lambda row: compute_weight_factor(
                row["mean_inv_sigma_crit"],
                row["i_hsmshaperegauss_derived_sigma_e"],
                row["i_hsmshaperegauss_derived_rms_e"],
            ),
            axis=1,
        )

        mult_correction = compute_mult_bias_correction(bin_data)
        responsivity = compute_responsivity_correction(bin_data)

        effective_radius_values.append(compute_effective_radius(bin_data))
        delta_sigma_values.append(
            compute_delta_sigma(responsivity, mult_correction, bin_data)
        )
        var_shape_values.append(
            compute_var_shape(responsivity, mult_correction, bin_data)
        )

        w_sum = bin_data["w"].sum()
        bin_mean_inv_sigma_crit_values.append(
            (bin_data["w"] * bin_data["mean_inv_sigma_crit"]).sum() / w_sum
        )

    delta_sigma_arr      = np.array(delta_sigma_values)
    radius_arr           = np.array(effective_radius_values)
    bin_sigma_crit_arr   = np.array(bin_mean_inv_sigma_crit_values)
    var_shape_arr        = np.array(var_shape_values)

    # Drop empty bins: any NaN in the variance propagates to a NaN diagonal in
    # the inverse covariance and causes the likelihood to return NaN (not -inf),
    # which crashes emcee.  Bins with no galaxies carry no information and
    # should simply be excluded from the analysis.
    valid = np.isfinite(var_shape_arr)
    inverse_covariance_matrix = np.diag(1.0 / var_shape_arr[valid])

    return (
        delta_sigma_arr[valid],
        radius_arr[valid],
        bin_sigma_crit_arr[valid],
        inverse_covariance_matrix,
    )


def get_theory_vector_binned_sigma(
    radius: NDArray[np.float64],
    bin_mean_inv_sigma_crit: NDArray[np.float64],
    mset: Ncm.MSet,
) -> NDArray[np.float64]:
    """
    Compute the theoretical prediction for the excess surface mass density (Delta
    Sigma) at given radii, including the reduced-shear correction (Eq. 22).

    Parameters
    ----------
    radius : NDArray[np.float64]
        The radii at which to compute the theoretical prediction.
    bin_mean_inv_sigma_crit : NDArray[np.float64]
        The per-bin weight-averaged
        $\\langle \\langle \\Sigma^{-1}_{\\mathrm{cr}} \\rangle \\rangle$ from
        get_data_vector_binned_sigma, used for
        the next-to-leading-order reduced-shear correction.
    mset : Ncm.MSet
        The cosmological model to use for the calculation.

    Returns
    -------
    NDArray[np.float64]
        The theoretical prediction for Delta Sigma at the given radii.
    """
    smd = cast(Nc.WLSurfaceMassDensity, mset.peek_by_name("NcWLSurfaceMassDensity"))
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    dp = cast(Nc.HaloDensityProfile, mset.peek_by_name("NcHaloDensityProfile"))

    z_cl = hp["z"]
    delta_sigma_nfw = smd.sigma_excess_array(dp, cosmo, radius, 1.0, 1.0, z_cl)
    sigma_nfw = smd.sigma_array(dp, cosmo, radius, 1.0, 1.0, z_cl)

    # Next-to-leading-order reduced-shear correction (Eq. 22)
    return delta_sigma_nfw / (1.0 - bin_mean_inv_sigma_crit * sigma_nfw)


def ln_likelihood_binned_sigma(
    theta: NDArray[np.float64],
    fparams: Sequence[str],
    fparams_bounds: Sequence[Tuple[float, float]] | None,
    radius: NDArray[np.float64],
    bin_mean_inv_sigma_crit: NDArray[np.float64],
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

    theory_vector = get_theory_vector_binned_sigma(
        radius, bin_mean_inv_sigma_crit, mset
    )
    if not np.all(np.isfinite(theory_vector)):
        return -np.inf

    diff = data_vector - theory_vector
    result = -0.5 * float(diff.T @ inv_cov @ diff)
    return result if np.isfinite(result) else -np.inf


# Per-process cache for mean_inv_sigma_crit: keyed by (id(pz_spline), z_cluster).
# Safe because splines pre-built in GalaxyProperties are stable objects within a process,
# z_cluster encodes the cosmology+position dependence, and cosmology is fixed per run.
_MISC_CACHE: dict = {}


def _mean_inv_sigma_crit_cached(pz_spline, mset: Ncm.MSet, z_cluster: float) -> float:
    """Memoized compute_mean_inv_sigma_critical: identical result, computed once per (galaxy, z_cluster)."""
    key = (id(pz_spline), z_cluster)
    if key not in _MISC_CACHE:
        _MISC_CACHE[key] = compute_mean_inv_sigma_critical(pz_spline, mset)
    return _MISC_CACHE[key]


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
    lp = ln_likelihood_binned_sigma(
        theta,
        pre["fparams"],
        pre["fparams_bounds"],
        pre["radius"],
        pre["bin_mean_inv_sigma_crit"],
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


class BinnedSigmaLikelihood(Likelihood):
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
    _bin_mean_inv_sigma_crit: NDArray[np.float64] = PrivateAttr()
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
            self._bin_mean_inv_sigma_crit,
            self._inv_cov,
        ) = get_data_vector_binned_sigma(self.bin_edges, self.obs, self.mset)

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
            v = -ln_likelihood_binned_sigma(
                theta,
                self.fparams,
                self.fparams_bounds,
                self._radius,
                self._bin_mean_inv_sigma_crit,
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
        filename: str = "binned_sigma_mcmc.h5",
    ) -> pd.DataFrame:
        """Run emcee with per-process mset reconstruction."""

        pre = {
            "fparams": list(self.fparams),
            "fparams_bounds": (
                list(self.fparams_bounds) if self.fparams_bounds is not None else None
            ),
            "data_vector": self._data_vector,
            "radius": self._radius,
            "bin_mean_inv_sigma_crit": self._bin_mean_inv_sigma_crit,
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
            # Fork context: workers inherit state instead of re-importing
            # (Python 3.14 defaults to forkserver, which breaks here).
            with get_context("fork").Pool(
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
