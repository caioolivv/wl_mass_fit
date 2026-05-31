"""Tests for the wl-mass-fit CLI."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli.app import app
from src.mock.mset import create_mock_mset
from src.mock.generator import MockGenerator
from src.mock.source import DataFrameGalaxySource

runner = CliRunner()


def _make_obs_parquet(tmp_path: Path) -> Path:
    """Generate a small mock obs DataFrame and save to parquet."""
    n = 20
    mset = create_mock_mset(ra=0.0, dec=0.0, z=0.3, log10MDelta=14.5)
    pz_nodes = np.linspace(0.0, 1.0, 11)
    pz_weights = np.array([0.0, 0.02, 0.05, 0.15, 0.3, 0.3, 0.15, 0.05, 0.02, 0.0, 0.0])
    pz_weights /= pz_weights.sum()

    source_df = pd.DataFrame({
        "z": np.full(n, 0.4),
        "pz_nodes": [pz_nodes.copy() for _ in range(n)],
        "pz_weights": [pz_weights.copy() for _ in range(n)],
        "i_hsmshaperegauss_derived_shear_bias_m": np.zeros(n),
        "i_hsmshaperegauss_derived_shear_bias_c1": np.zeros(n),
        "i_hsmshaperegauss_derived_shear_bias_c2": np.zeros(n),
        "i_hsmshaperegauss_derived_sigma_e": np.full(n, 0.3),
        "i_hsmshaperegauss_derived_rms_e": np.full(n, 0.1),
    })

    source = DataFrameGalaxySource([(0.3, source_df)])
    generator = MockGenerator(
        mset=mset,
        source=source,
        true_ra=0.0,
        true_dec=0.0,
        r_min=0.01,
        r_max=3.0,
        r_miscenter=0.0,
        n_gals=10,
        p_cut=0.3,
        delta_z=0.1,
    )

    from numcosmo_py import Ncm
    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    rng = Ncm.RNG.seeded_new(None, 42)
    obs = generator.generate(rng)

    parquet_path = tmp_path / "obs.parquet"
    obs.to_parquet(parquet_path, index=False)
    return parquet_path


def test_help():
    """wl-mass-fit --help exits cleanly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "mc-mle" in result.output
    assert "mc-posterior" in result.output
    assert "fit-mle" in result.output
    assert "fit-posterior" in result.output


def test_mc_mle_help():
    """mc-mle --help shows required flags."""
    result = runner.invoke(app, ["mc-mle", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--n-iter" in result.output
    assert "--true-ra" in result.output
    assert "--clusters-path" in result.output


def test_mc_posterior_help():
    """mc-posterior --help shows MCMC-specific flags."""
    result = runner.invoke(app, ["mc-posterior", "--help"])
    assert result.exit_code == 0
    assert "--nsamples" in result.output
    assert "--nwalkers" in result.output
    assert "--burn-in" in result.output


def test_fit_mle_help():
    """fit-mle --help shows --input."""
    result = runner.invoke(app, ["fit-mle", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--likelihood" in result.output


def test_fit_posterior_help():
    """fit-posterior --help shows --nsamples."""
    result = runner.invoke(app, ["fit-posterior", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--nsamples" in result.output


def test_fit_mle_runs(tmp_path):
    """fit-mle succeeds on a generated obs parquet and prints result."""
    obs_path = _make_obs_parquet(tmp_path)
    result = runner.invoke(app, [
        "fit-mle",
        "--input", str(obs_path),
        "--likelihood", "binned-shear",
    ])
    assert result.exit_code == 0, result.output
    assert "log10MDelta" in result.output


def test_fit_mle_output_parquet(tmp_path):
    """fit-mle --output writes a parquet with the fitted param."""
    obs_path = _make_obs_parquet(tmp_path)
    out_path = tmp_path / "result.parquet"
    result = runner.invoke(app, [
        "fit-mle",
        "--input", str(obs_path),
        "--output", str(out_path),
        "--likelihood", "binned-shear",
    ])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert "log10MDelta" in df.columns


def test_fit_posterior_runs(tmp_path):
    """fit-posterior runs a tiny chain and prints summary."""
    obs_path = _make_obs_parquet(tmp_path)
    out_path = tmp_path / "chain.parquet"
    result = runner.invoke(app, [
        "fit-posterior",
        "--input", str(obs_path),
        "--nsamples", "20",
        "--nwalkers", "4",
        "--burn-in", "2",
        "--output", str(out_path),
        "--likelihood", "binned-shear",
        "--no-progress",
    ])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    chain = pd.read_parquet(out_path)
    assert "log10MDelta" in chain.columns
    assert len(chain) > 0
