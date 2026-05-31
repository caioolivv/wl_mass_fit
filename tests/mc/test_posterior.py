"""Tests for MCPosterior analysis."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.mock.generator import MockGenerator
from src.mock.source import DataFrameGalaxySource
from src.mock.mset import create_mock_mset
from src.likelihood.binned_shear import BinnedShearLikelihood
from src.likelihood.mset import create_mset
from src.mc.posterior import MCPosterior
from src.utils.utils import CoordSystem


@pytest.fixture
def mock_obs_df():
    """Create a minimal observation DataFrame for testing."""
    n = 20
    pz_nodes = np.linspace(0.0, 1.0, 11)
    pz_weights = np.array([0.0, 0.02, 0.05, 0.15, 0.3, 0.3, 0.15, 0.05, 0.02, 0.0, 0.0])
    pz_weights /= pz_weights.sum()

    return pd.DataFrame({
        "z": np.full(n, 0.4),
        "pz_nodes": [pz_nodes.copy() for _ in range(n)],
        "pz_weights": [pz_weights.copy() for _ in range(n)],
        "i_hsmshaperegauss_derived_shear_bias_m": np.zeros(n),
        "i_hsmshaperegauss_derived_shear_bias_c1": np.zeros(n),
        "i_hsmshaperegauss_derived_shear_bias_c2": np.zeros(n),
        "i_hsmshaperegauss_derived_sigma_e": np.full(n, 0.3),
        "i_hsmshaperegauss_derived_rms_e": np.full(n, 0.1),
    })


@pytest.fixture
def mock_mset():
    """Create a mock mset for testing."""
    return create_mock_mset(ra=0.0, dec=0.0, z=0.3, log10MDelta=14.5)


@pytest.fixture
def fit_mset():
    """Separate fit mset — must not be shared with the generator's mset."""
    return create_mset(ra=0.0, dec=0.0, z=0.3, log10MDelta=14.5)


@pytest.fixture
def mock_generator(mock_mset, mock_obs_df):
    """Create a MockGenerator for testing."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    return MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.01,
        r_max=3.0,
        r_miscenter=0.0,
        n_gals=5,
        p_cut=0.3,
        delta_z=0.1,
    )


def likelihood_factory(obs_df, mset):
    """Create a BinnedShearLikelihood from obs DataFrame."""
    return BinnedShearLikelihood(
        mset=mset,
        obs=obs_df,
        coord_system=CoordSystem.CELESTIAL,
        fparams=["NcHaloMassSummary:log10MDelta"],
    )


def test_mcposterior_creation(mock_generator, fit_mset):
    """Test MCPosterior instantiation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCPosterior(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=2,
            output=output,
            nsamples=100,
            nwalkers=16,
            nthreads=1,
            burn_in=10,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )

        assert mc.n_iter == 2
        assert mc.nsamples == 100
        assert mc.nwalkers == 16
        assert mc.burn_in == 10


def test_mcposterior_run(mock_generator, fit_mset):
    """Test MCPosterior.run() with small n_iter and nsamples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCPosterior(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=1,  # Just 1 to keep test fast
            output=output,
            nsamples=20,
            nwalkers=4,
            nthreads=1,
            burn_in=2,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
            progress=False,
        )

        df = mc.run()

        assert len(df) >= 1
        assert "fit_time" in df.columns
        assert "success" in df.columns
        # Check for posterior summary columns
        assert any("_mean" in col for col in df.columns)
        assert any("_std" in col for col in df.columns)
        assert output.exists()


def test_mcposterior_summary(mock_generator, fit_mset):
    """Test MCPosterior.summary() statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCPosterior(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=1,
            output=output,
            nsamples=20,
            nwalkers=4,
            nthreads=1,
            burn_in=2,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
            progress=False,
        )

        df = mc.run()
        result = mc.summary()

        assert result.n_total > 0
        assert result.n_successful > 0
        assert 0 <= result.success_rate <= 1
        # MCPosterior strips the model prefix from param names
        assert "log10MDelta" in result.per_param
        param_stats = result.per_param["log10MDelta"]
        assert "bias_mean" in param_stats
        assert "spread_mean" in param_stats
        assert "uncertainty_std" in param_stats
        assert "coverage_1sigma" in param_stats
