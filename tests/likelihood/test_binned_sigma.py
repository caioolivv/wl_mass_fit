"""
Test suite for the binned_sigma module.
"""

from src.likelihood.binned_sigma import (
    BinnedSigmaLikelihood,
    compute_effective_radius,
    compute_mean_inv_sigma_critical,
    compute_mult_bias_correction,
    compute_responsivity_correction,
    compute_weight_factor,
    get_theory_vector_binned_sigma,
    ln_likelihood_binned_sigma,
)
from src.likelihood.mset import create_mset
from src.utils.utils import (
    compute_cross_component,
    compute_tangential_component,
    create_ncm_spline,
)

from typing import cast
import numpy as np
import pandas as pd
import pytest
from numcosmo_py import Ncm, Nc

Ncm.cfg_init()
Ncm.cfg_set_log_handler(lambda msg: None)


N_GAL = 80
BIN_EDGES = [0.5, 1.0, 1.5, 2.0, 2.5]
Z_SRC_MEAN = 0.8
RNG = np.random.default_rng(42)
FPARAM = "NcHaloMassSummary:log10MDelta"


def _build_obs(n_gal: int = N_GAL) -> pd.DataFrame:
    """Build a DataFrame satisfying Likelihood.REQUIRED_COLUMNS."""
    nodes = np.linspace(0.01, 3.0, 50)
    pz = np.exp(-0.5 * ((nodes - Z_SRC_MEAN) / 0.1) ** 2)
    return pd.DataFrame(
        {
            "i_ra": RNG.uniform(-0.05, 0.05, n_gal),
            "i_dec": RNG.uniform(-0.05, 0.05, n_gal),
            "i_hsmshaperegauss_e1": RNG.uniform(-0.3, 0.3, n_gal),
            "i_hsmshaperegauss_e2": RNG.uniform(-0.3, 0.3, n_gal),
            "i_hsmshaperegauss_derived_rms_e": np.full(n_gal, 0.26),
            "i_hsmshaperegauss_derived_sigma_e": np.full(n_gal, 0.05),
            "i_hsmshaperegauss_derived_shear_bias_m": np.full(n_gal, 0.01),
            "i_hsmshaperegauss_derived_shear_bias_c1": np.zeros(n_gal),
            "i_hsmshaperegauss_derived_shear_bias_c2": np.zeros(n_gal),
            "pz_weights": [pz for _ in range(n_gal)],
            "pz_nodes": [nodes for _ in range(n_gal)],
        }
    )


@pytest.fixture(name="mset")
def fixture_mset() -> Ncm.MSet:
    return create_mset()


@pytest.fixture(name="likelihood")
def fixture_likelihood(mset: Ncm.MSet) -> BinnedSigmaLikelihood:
    lk = BinnedSigmaLikelihood(mset=mset, obs=_build_obs(), bin_edges=BIN_EDGES)
    lk.prepare_data()
    return lk


# --- pure helper functions -------------------------------------------------


def test_compute_mean_inv_sigma_critical_value():
    pz_nodes = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    pz_weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    expected = (
        1.0 / 0.8
    )  # Assuming lens at z=0.5, Sigma_crit ~ 1/(1 - D_lens/D_source) ~ 1/(1 - 0.5/0.8) = 1/0.375 = 1.0 / 0.8
    assert compute_mean_inv_sigma_critical(
        create_ncm_spline(pz_weights, pz_nodes),
        mset=create_mset(),
    ) == pytest.approx(expected)


def test_compute_weight_factor_matches_formula():
    mean_inv = np.array([1.0, 2.0])
    sigma_err = np.array([0.1, 0.2])
    sigma_rms = np.array([0.2, 0.4])
    expected = mean_inv**2 / (sigma_err**2 + sigma_rms**2)
    np.testing.assert_allclose(
        compute_weight_factor(mean_inv, sigma_err, sigma_rms), expected
    )


def test_compute_mult_bias_correction_value():
    df = pd.DataFrame(
        {
            "w": [1.0, 1.0],
            "i_hsmshaperegauss_derived_shear_bias_m": [0.0, 0.02],
        }
    )
    assert compute_mult_bias_correction(df) == pytest.approx(1.01)


def test_compute_mult_bias_correction_zero_weight_is_nan():
    df = pd.DataFrame(
        {"w": [0.0, 0.0], "i_hsmshaperegauss_derived_shear_bias_m": [0.01, 0.02]}
    )
    assert np.isnan(compute_mult_bias_correction(df))


def test_compute_responsivity_correction_value():
    df = pd.DataFrame({"w": [1.0, 1.0], "i_hsmshaperegauss_derived_rms_e": [0.2, 0.3]})
    expected = 1 - (1.0 * 0.04 + 1.0 * 0.09) / 2.0
    assert compute_responsivity_correction(df) == pytest.approx(expected)


def test_compute_responsivity_correction_zero_weight_is_nan():
    df = pd.DataFrame({"w": [0.0], "i_hsmshaperegauss_derived_rms_e": [0.26]})
    assert np.isnan(compute_responsivity_correction(df))


def test_compute_effective_radius_value():
    df = pd.DataFrame({"w": [1.0, 1.0], "radius": [1.0, 2.0]})
    assert compute_effective_radius(df) == pytest.approx(2.0 / 1.5)


def test_compute_effective_radius_zero_weight_is_nan():
    df = pd.DataFrame({"w": [0.0, 0.0], "radius": [1.0, 2.0]})
    assert np.isnan(compute_effective_radius(df))


# --- prepared data and theory vectors --------------------------------------


def test_prepare_data_populates_arrays(likelihood: BinnedSigmaLikelihood):
    n_bins = len(BIN_EDGES) - 1
    assert likelihood._data_vector.shape == (n_bins,)
    assert likelihood._radius.shape == (n_bins,)
    assert likelihood._bin_mean_inv_sigma_crit.shape == (n_bins,)
    assert likelihood._inv_cov.shape == (n_bins, n_bins)


def test_theory_vector_shape_and_finite(
    mset: Ncm.MSet, likelihood: BinnedSigmaLikelihood
):
    theory = get_theory_vector_binned_sigma(
        likelihood._radius, likelihood._bin_mean_inv_sigma_crit, mset
    )
    assert theory.shape == likelihood._radius.shape
    assert np.all(np.isfinite(theory))


def test_theory_vector_increases_with_mass(
    mset: Ncm.MSet, likelihood: BinnedSigmaLikelihood
):
    """At fixed cosmology, Delta Sigma increases monotonically with halo mass."""
    hms = cast(Nc.HaloMassSummary, mset.peek_by_name("NcHaloMassSummary"))
    hms["log10MDelta"] = 14.0
    low = get_theory_vector_binned_sigma(
        likelihood._radius, likelihood._bin_mean_inv_sigma_crit, mset
    )
    hms["log10MDelta"] = 14.8
    high = get_theory_vector_binned_sigma(
        likelihood._radius, likelihood._bin_mean_inv_sigma_crit, mset
    )
    assert np.all(high > low)


# --- ln_likelihood_binned_sigma --------------------------------------------


def _ln_args(likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet, bounds):
    return (
        [FPARAM],
        bounds,
        likelihood._radius,
        likelihood._bin_mean_inv_sigma_crit,
        likelihood._data_vector,
        likelihood._inv_cov,
        mset,
    )


def test_ln_likelihood_finite_within_bounds(
    likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet
):
    lp = ln_likelihood_binned_sigma(
        np.array([14.5]), *_ln_args(likelihood, mset, [(13.0, 16.0)])
    )
    assert np.isfinite(lp)
    assert lp <= 0.0


def test_ln_likelihood_outside_bounds_is_neg_inf(
    likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet
):
    lp = ln_likelihood_binned_sigma(
        np.array([20.0]), *_ln_args(likelihood, mset, [(13.0, 16.0)])
    )
    assert lp == -np.inf


def test_ln_likelihood_none_bounds_disables_check(
    likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet
):
    lp = ln_likelihood_binned_sigma(np.array([14.5]), *_ln_args(likelihood, mset, None))
    assert np.isfinite(lp)


def test_ln_likelihood_varies_with_theta(
    likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet
):
    args = _ln_args(likelihood, mset, [(13.0, 16.0)])
    lp1 = ln_likelihood_binned_sigma(np.array([14.0]), *args)
    lp2 = ln_likelihood_binned_sigma(np.array([14.5]), *args)
    assert lp1 != lp2


def test_ln_likelihood_writes_theta_into_mset(
    likelihood: BinnedSigmaLikelihood, mset: Ncm.MSet
):
    """ln_likelihood_binned_sigma should set the named fparam on the mset."""
    ln_likelihood_binned_sigma(
        np.array([14.7]), *_ln_args(likelihood, mset, [(13.0, 16.0)])
    )
    hms = cast(Nc.HaloMassSummary, mset.peek_by_name("NcHaloMassSummary"))
    assert hms["log10MDelta"] == pytest.approx(14.7)


# --- BinnedSigmaLikelihood high-level --------------------------------------


def test_maximum_likelihood_estimate_shape(likelihood: BinnedSigmaLikelihood):
    mle = likelihood.maximum_likelihood_estimate()
    assert mle.shape == (len(likelihood.fparams),)
    assert np.all(np.isfinite(mle))


def test_validate_bin_edges_rejects_non_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        BinnedSigmaLikelihood(
            mset=create_mset(), obs=_build_obs(), bin_edges=[1.0, 0.5, 2.0]
        )


def test_validate_bin_edges_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        BinnedSigmaLikelihood(
            mset=create_mset(), obs=_build_obs(), bin_edges=[-0.1, 1.0, 2.0]
        )
