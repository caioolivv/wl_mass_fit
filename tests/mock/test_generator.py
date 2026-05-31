"""
Tests for mock generation.
"""

import pytest
import numpy as np
import pandas as pd
from numcosmo_py import Ncm

from src.mock.generator import MockGenerator
from src.mock.source import DataFrameGalaxySource, GalaxyProperties
from src.mock.mset import create_mock_mset
from src.utils.utils import CoordSystem


@pytest.fixture
def mock_obs_df():
    """Create a minimal observation DataFrame for testing."""
    n = 20
    pz_nodes = np.linspace(0.0, 1.0, 11)
    # P(z) with peak at z~0.4-0.5
    pz_weights = np.array([0.0, 0.02, 0.05, 0.15, 0.3, 0.3, 0.15, 0.05, 0.02, 0.0, 0.0])
    pz_weights /= pz_weights.sum()  # Normalize

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
def mock_generator(mock_mset, mock_obs_df):
    """Create a MockGenerator for testing."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    return MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.0,
        n_gals=10,
        p_cut=0.5,  # Relaxed cut for testing
        delta_z=0.05,  # Smaller delta for testing
    )


def test_mock_generator_creation(mock_generator):
    """Test MockGenerator initialization."""
    assert mock_generator.n_gals == 10
    assert mock_generator.true_ra == 0.0
    assert mock_generator.true_dec == 0.0
    assert mock_generator.r_min == 0.1
    assert mock_generator.r_max == 2.0


def test_mock_generator_validation_negative_n_gals(mock_mset, mock_obs_df):
    """Test that negative n_gals raises ValueError."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    with pytest.raises(ValueError, match="n_gals must be positive"):
        MockGenerator(
            mset=mock_mset,
            source=source,
            true_ra=0.0,
            true_dec=0.0,
            r_min=0.1,
            r_max=2.0,
            r_miscenter=0.0,
            n_gals=-1,
        )


def test_mock_generator_validation_negative_radius(mock_mset, mock_obs_df):
    """Test that negative radii raise ValueError."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    with pytest.raises(ValueError, match="Radii must be non-negative"):
        MockGenerator(
            mset=mock_mset,
            source=source,
            true_ra=0.0,
            true_dec=0.0,
            r_min=-0.1,
            r_max=2.0,
            r_miscenter=0.0,
            n_gals=10,
        )


def test_mock_generator_generate(mock_generator):
    """Test basic mock generation."""
    rng = Ncm.RNG.new()
    df = mock_generator.generate(rng)

    # Check DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == mock_generator.n_gals

    # Check required columns
    required_cols = {
        "i_ra", "i_dec",
        "i_hsmshaperegauss_e1", "i_hsmshaperegauss_e2",
        "i_hsmshaperegauss_derived_rms_e",
        "i_hsmshaperegauss_derived_sigma_e",
        "i_hsmshaperegauss_derived_shear_bias_m",
        "i_hsmshaperegauss_derived_shear_bias_c1",
        "i_hsmshaperegauss_derived_shear_bias_c2",
        "z", "pz_nodes", "pz_weights",
    }
    assert required_cols.issubset(set(df.columns))


def test_mock_generator_radii_in_range(mock_mset, mock_obs_df):
    """Test that generated galaxies are within radius bounds."""
    from src.utils.utils import compute_radius

    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    generator = MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.0,
        n_gals=3,
        p_cut=0.5,
        delta_z=0.05,
    )

    rng = Ncm.RNG.new()
    rng.set_seed(42)
    df = generator.generate(rng)

    cosmo = generator.mset.peek_by_name("NcHICosmo")
    h = cosmo.h()

    # Compute radius in Mpc from RA/Dec
    radii = df.apply(
        lambda row: compute_radius(row["i_ra"], row["i_dec"], generator.mset),
        axis=1
    )

    r_min_mpc = generator.r_min / h
    r_max_mpc = generator.r_max / h

    assert (radii >= r_min_mpc - 0.01).all()  # Small tolerance for rounding
    assert (radii <= r_max_mpc + 0.01).all()


def test_mock_generator_redshift_bounds(mock_mset, mock_obs_df):
    """Test that generated redshifts pass photo-z cut."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    generator = MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.0,
        n_gals=3,
        z_max=2.5,
        p_cut=0.5,
        delta_z=0.05,
    )

    rng = Ncm.RNG.new()
    rng.set_seed(42)
    df = generator.generate(rng)

    assert (df["z"] <= generator.z_max).all()


def test_mock_generator_pz_arrays(mock_mset, mock_obs_df):
    """Test that pz_nodes and pz_weights are arrays."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    generator = MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.0,
        n_gals=3,
        p_cut=0.5,
        delta_z=0.05,
    )

    rng = Ncm.RNG.new()
    rng.set_seed(42)
    df = generator.generate(rng)

    for idx, row in df.iterrows():
        assert isinstance(row["pz_nodes"], np.ndarray)
        assert isinstance(row["pz_weights"], np.ndarray)
        assert len(row["pz_nodes"]) > 0
        assert len(row["pz_weights"]) == len(row["pz_nodes"])


def test_mock_generator_determinism(mock_generator):
    """Test that same seed produces same DataFrame (up to row order)."""
    rng1 = Ncm.RNG.new()
    rng1.set_seed(12345)
    df1 = mock_generator.generate(rng1)

    rng2 = Ncm.RNG.new()
    rng2.set_seed(12345)
    df2 = mock_generator.generate(rng2)

    # Should be identical
    assert len(df1) == len(df2)
    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_mock_generator_with_miscentering(mock_mset, mock_obs_df):
    """Test generation with non-zero miscentering."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    generator = MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.2,  # Non-zero
        n_gals=3,
        p_cut=0.5,
        delta_z=0.05,
    )

    rng = Ncm.RNG.new()
    df = generator.generate(rng)

    assert len(df) == 3
    assert "i_ra" in df.columns


def test_mock_generator_euclidean_coords(mock_mset, mock_obs_df):
    """Test generation with Euclidean coordinate system."""
    source = DataFrameGalaxySource([(0.3, mock_obs_df)])
    generator = MockGenerator(
        mset=mock_mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.1,
        r_max=2.0,
        r_miscenter=0.0,
        n_gals=3,
        p_cut=0.5,
        delta_z=0.05,
        coord_system=CoordSystem.EUCLIDEAN,
    )

    rng = Ncm.RNG.new()
    df = generator.generate(rng)

    assert len(df) == 3


def test_mock_generator_mset_validation(mock_obs_df):
    """Test that mset validation catches missing required submodels."""
    from numcosmo_py import Ncm

    # Create an mset without required models
    bad_mset = Ncm.MSet.new_array([])

    source = DataFrameGalaxySource([(0.3, mock_obs_df)])

    with pytest.raises(ValueError, match="NcHICosmo"):
        MockGenerator(
            mset=bad_mset,
            source=source,
            true_ra=0.0,
            true_dec=0.0,
            r_min=0.1,
            r_max=2.0,
            r_miscenter=0.0,
            n_gals=10,
        )
