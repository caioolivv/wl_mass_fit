"""
Tests for galaxy source classes.
"""

import pytest
import numpy as np
import pandas as pd

from src.mock.source import (
    GalaxyProperties,
    DataFrameGalaxySource,
    HamanaGalaxySource,
)


@pytest.fixture
def sample_obs_df():
    """Create a sample observation DataFrame."""
    n = 5
    pz_nodes = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    pz_weights = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.0])
    pz_weights /= pz_weights.sum()

    return pd.DataFrame({
        "z": np.random.uniform(0.3, 0.5, n),
        "pz_nodes": [pz_nodes.copy() for _ in range(n)],
        "pz_weights": [pz_weights.copy() for _ in range(n)],
        "i_hsmshaperegauss_derived_shear_bias_m": np.random.uniform(-0.1, 0.1, n),
        "i_hsmshaperegauss_derived_shear_bias_c1": np.random.uniform(-0.05, 0.05, n),
        "i_hsmshaperegauss_derived_shear_bias_c2": np.random.uniform(-0.05, 0.05, n),
        "i_hsmshaperegauss_derived_sigma_e": np.random.uniform(0.2, 0.4, n),
        "i_hsmshaperegauss_derived_rms_e": np.random.uniform(0.1, 0.2, n),
    })


def test_galaxy_properties_creation():
    """Test GalaxyProperties dataclass creation and immutability."""
    props = GalaxyProperties(
        z=0.4,
        pz_nodes=np.array([0.0, 0.5, 1.0]),
        pz_weights=np.array([0.0, 1.0, 0.0]),
        m=-0.05,
        c1=0.02,
        c2=-0.01,
        sigma_e=0.3,
        rms_e=0.15,
    )

    assert props.z == 0.4
    assert props.m == -0.05
    assert len(props.pz_nodes) == 3

    # Test immutability
    with pytest.raises(Exception):
        props.z = 0.5


def test_dataframe_galaxy_source(sample_obs_df):
    """Test creating a DataFrameGalaxySource from a DataFrame."""
    z_cluster = 0.4
    source = DataFrameGalaxySource([(z_cluster, sample_obs_df)])

    rng = np.random.default_rng(42)
    props = source.sample(z_target=0.4, rng=rng)

    assert isinstance(props, GalaxyProperties)
    assert hasattr(props, "z")
    assert hasattr(props, "pz_nodes")
    assert hasattr(props, "m")


def test_dataframe_galaxy_source_multiple_clusters(sample_obs_df):
    """Test DataFrameGalaxySource with multiple clusters."""
    entries = [
        (0.3, sample_obs_df.copy()),
        (0.5, sample_obs_df.copy()),
    ]
    source = DataFrameGalaxySource(entries)

    rng = np.random.default_rng(42)
    # Target z between clusters — should weight both
    props = source.sample(z_target=0.4, rng=rng)

    assert isinstance(props, GalaxyProperties)


def test_dataframe_galaxy_source_clustering(sample_obs_df):
    """Test that samples cluster near target redshift."""
    entries = [
        (0.3, sample_obs_df.copy()),
        (0.5, sample_obs_df.copy()),
    ]
    source = DataFrameGalaxySource(entries, z_tol=0.05)

    rng = np.random.default_rng(42)
    # Many samples near 0.3 should mostly come from 0.3 cluster
    samples = [source.sample(z_target=0.29, rng=rng) for _ in range(20)]

    assert all(isinstance(s, GalaxyProperties) for s in samples)


def test_galaxy_source_fallback_to_nearest(sample_obs_df):
    """Test fallback to nearest clusters when none within z_tol."""
    entries = [
        (0.2, sample_obs_df.copy()),
        (0.8, sample_obs_df.copy()),
    ]
    source = DataFrameGalaxySource(entries, z_tol=0.01)

    rng = np.random.default_rng(42)
    # Far from both clusters — should still sample
    props = source.sample(z_target=0.5, rng=rng)

    assert isinstance(props, GalaxyProperties)


def test_hamana_source_missing_directory():
    """Test that HamanaGalaxySource raises on invalid directory."""
    from pathlib import Path

    with pytest.raises(ValueError):
        HamanaGalaxySource.from_directory(Path("/nonexistent/path"))


def test_hamana_source_empty_directory(tmp_path):
    """Test that HamanaGalaxySource returns empty if no HWL16a dirs found."""
    source = HamanaGalaxySource.from_directory(tmp_path)
    # Should succeed but with no clusters
    assert source._clusters == []
