"""
Test suite for Likelihood.
"""

from src.likelihood.likelihood import Likelihood
from src.likelihood.binned_sigma import BinnedSigmaLikelihood
from src.likelihood.mset import create_mset
from numcosmo_py import Ncm, Nc
from typing import cast
import numpy as np
import pandas as pd
import pytest


def test_likelihood_is_abstract():
    """Test that Likelihood cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Likelihood(obs=pd.DataFrame())  # pyright: ignore[reportAbstractUsage]


def test_fparams_configuration_uses_instance_mset():
    obs = pd.DataFrame(
        {
            "i_ra": [0.0],
            "i_dec": [0.0],
            "i_hsmshaperegauss_e1": [0.0],
            "i_hsmshaperegauss_e2": [0.0],
            "i_hsmshaperegauss_derived_rms_e": [0.26],
            "i_hsmshaperegauss_derived_sigma_e": [0.05],
            "i_hsmshaperegauss_derived_shear_bias_m": [0.0],
            "i_hsmshaperegauss_derived_shear_bias_c1": [0.0],
            "i_hsmshaperegauss_derived_shear_bias_c2": [0.0],
            "z": [0.3],
            "pz_weights": [np.array([1.0])],
            "pz_nodes": [np.array([0.3])],
        }
    )

    likelihood = BinnedSigmaLikelihood(
        mset=create_mset(),
        obs=obs,
        bin_edges=[0.5, 1.0],
        fparams=["NcHICosmo:H0"],
        fparams_bounds=[(60.0, 80.0)],
    )

    cosmo = cast(Nc.HICosmo, likelihood.mset.peek_by_name("NcHICosmo"))
    hms = cast(Nc.HaloMassSummary, likelihood.mset.peek_by_name("NcHaloMassSummary"))

    assert cosmo.param_get_desc("H0").get("fit") is True
    assert cosmo.param_get_desc("H0").get("lower-bound") == pytest.approx(60.0)
    assert cosmo.param_get_desc("H0").get("upper-bound") == pytest.approx(80.0)
    assert hms.param_get_desc("log10MDelta").get("fit") is False
