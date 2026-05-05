"""
Test suite for the Obs Class.
"""

from src.obs.obs import Obs
from typing import Any
from enum import StrEnum
import pandas as pd
import pytest


class ValidTestObs(StrEnum):
    """
    Test cases for the Obs class.
    """

    VALID_DF = "valid_df"
    VALID_DF_EXTRA_COL = "valid_df_extra_col"


class InvalidTestObs(StrEnum):
    """
    Test cases for invalid Obs data.
    """

    VALID_DF_MISSING_COL = "valid_df_missing_col"
    NOT_A_DF = "not_a_df"
    NONE = "none"


@pytest.fixture(name="valid_obs_data", params=[e.value for e in ValidTestObs])
def fixture_valid_obs_data(request) -> pd.DataFrame:
    """
    Fixture to provide different valid DataFrames for testing the Obs class. The
    DataFrames will have the required columns, and some will have extra columns to test
    the validation logic.
    """
    if request.param == ValidTestObs.VALID_DF:
        data = {
            "i_ra": [1.0, 2.0],
            "i_dec": [3.0, 4.0],
            "i_hsmshaperegauss_e1": [0.1, 0.2],
            "i_hsmshaperegauss_e2": [0.3, 0.4],
            "i_hsmshaperegauss_derived_rms_e": [0.01, 0.02],
            "i_hsmshaperegauss_derived_sigma_e": [0.001, 0.002],
            "i_hsmshaperegauss_derived_shear_bias_m": [0.0001, 0.0002],
            "i_hsmshaperegauss_derived_shear_bias_c1": [0.00001, 0.00002],
            "i_hsmshaperegauss_derived_shear_bias_c2": [0.000001, 0.000002],
            "pz_weights": [0.5, 0.5],
            "pz_nodes": [[0.1, 0.2], [0.3, 0.4]],
        }
    elif request.param == ValidTestObs.VALID_DF_EXTRA_COL:
        data = {
            "i_ra": [1.0, 2.0],
            "i_dec": [3.0, 4.0],
            "i_hsmshaperegauss_e1": [0.1, 0.2],
            "i_hsmshaperegauss_e2": [0.3, 0.4],
            "i_hsmshaperegauss_derived_rms_e": [0.01, 0.02],
            "i_hsmshaperegauss_derived_sigma_e": [0.001, 0.002],
            "i_hsmshaperegauss_derived_shear_bias_m": [0.0001, 0.0002],
            "i_hsmshaperegauss_derived_shear_bias_c1": [0.00001, 0.00002],
            "i_hsmshaperegauss_derived_shear_bias_c2": [0.000001, 0.000002],
            "pz_weights": [0.5, 0.5],
            "pz_nodes": [[0.1, 0.2], [0.3, 0.4]],
            "extra_col": ["extra1", "extra2"],
        }
    else:  # pragma: no cover
        raise ValueError("Invalid test case")
    return pd.DataFrame(data)


@pytest.fixture(name="invalid_obs_data", params=[e.value for e in InvalidTestObs])
def fixture_invalid_obs_data(request) -> Any:
    """
    Fixture to provide different invalid data for testing the Obs class. This includes
    DataFrames missing required columns and non-DataFrame inputs.
    """
    if request.param == InvalidTestObs.VALID_DF_MISSING_COL:
        data = {
            "i_ra": [1.0, 2.0],
            "i_dec": [3.0, 4.0],
            # Missing required columns
        }
        return pd.DataFrame(data)
    elif request.param == InvalidTestObs.NOT_A_DF:
        return "This is not a DataFrame"
    elif request.param == InvalidTestObs.NONE:
        return None
    else:  # pragma: no cover
        raise ValueError("Invalid test case")


def test_obs_valid_data_instantiation(valid_obs_data: pd.DataFrame) -> None:
    """
    Test that the Obs class can be instantiated with valid data and that it correctly
    validates the required columns.
    """
    obs = Obs(data=valid_obs_data)
    assert isinstance(obs, Obs)
    assert obs.data.equals(valid_obs_data)


def test_obs_invalid_data_instantiation(invalid_obs_data: Any) -> None:
    """
    Test that the Obs class raises appropriate errors when instantiated with invalid data.
    This includes missing required columns and non-DataFrame inputs.
    """
    with pytest.raises(ValueError):
        Obs(data=invalid_obs_data)
