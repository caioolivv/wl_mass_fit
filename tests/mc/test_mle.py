"""Tests for MCMLE analysis."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from numcosmo_py import Ncm

from src.mock.generator import MockGenerator
from src.mock.source import DataFrameGalaxySource
from src.mock.mset import create_mock_mset
from src.likelihood.binned_shear import BinnedShearLikelihood
from src.likelihood.mset import create_mset
from src.mc.mle import MCMLE
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


def test_mcmle_creation(mock_generator, fit_mset):
    """Test MCMLE instantiation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCMLE(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=2,
            output=output,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )

        assert mc.n_iter == 2
        assert mc.seed == 42


def test_mcmle_run(mock_generator, fit_mset):
    """Test MCMLE.run() with small n_iter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCMLE(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=2,
            output=output,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )

        df = mc.run()

        assert len(df) >= 2  # At least 2 total iterations
        assert "fit_time" in df.columns
        assert "success" in df.columns
        assert "NcHaloMassSummary:log10MDelta" in df.columns
        assert output.exists()


def test_mcmle_resume(mock_generator, fit_mset):
    """Test MCMLE resumability."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        # First run: 2 iterations
        mc1 = MCMLE(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=2,
            output=output,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )
        df1 = mc1.run()
        n_first = len(df1)

        # Second run: ask for 4 total (should add 2 more)
        mc2 = MCMLE(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=4,
            output=output,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )
        df2 = mc2.run()
        n_second = len(df2)

        assert n_second >= n_first
        assert output.exists()


def test_mcmle_summary(mock_generator, fit_mset):
    """Test MCMLE.summary() statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "results.parquet"

        mc = MCMLE(
            generator=mock_generator,
            likelihood_factory=lambda obs: likelihood_factory(obs, fit_mset),
            n_iter=3,
            output=output,
            true_params={"NcHaloMassSummary:log10MDelta": 14.5},
            seed=42,
        )

        df = mc.run()
        result = mc.summary()

        assert result.n_total > 0
        assert result.n_successful > 0
        assert 0 <= result.success_rate <= 1
        assert "NcHaloMassSummary:log10MDelta" in result.per_param
        assert "mean" in result.per_param["NcHaloMassSummary:log10MDelta"]
        assert "std" in result.per_param["NcHaloMassSummary:log10MDelta"]
        assert "bias" in result.per_param["NcHaloMassSummary:log10MDelta"]
