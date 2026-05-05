"""
Unified data structure for shapeHSM observables, including P(z) information and weight
factors, based on Pandas DataFrames.
"""

from typing import Set, ClassVar
from pydantic import BaseModel, ConfigDict, Field, field_validator
import pandas as pd


class Obs(BaseModel):
    """
    A data structure to hold shapeHSM observables, including P(z) information and weight
    factors, based on Pandas DataFrames.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    data: pd.DataFrame = Field(
        ..., description="DataFrame containing shapeHSM observables."
    )

    REQUIRED_COLUMNS: ClassVar[Set[str]] = {
        "i_ra",
        "i_dec",
        "i_hsmshaperegauss_e1",
        "i_hsmshaperegauss_e2",
        "i_hsmshaperegauss_derived_rms_e",
        "i_hsmshaperegauss_derived_sigma_e",
        "i_hsmshaperegauss_derived_shear_bias_m",
        "i_hsmshaperegauss_derived_shear_bias_c1",
        "i_hsmshaperegauss_derived_shear_bias_c2",
        "pz_weights",
        "pz_nodes",
    }

    @field_validator("data")
    @classmethod
    def validate_required_columns(cls, v: pd.DataFrame) -> pd.DataFrame:
        if not cls.REQUIRED_COLUMNS.issubset(v.columns):
            missing_cols = cls.REQUIRED_COLUMNS - set(v.columns)
            raise ValueError(f"Missing required columns: {missing_cols}")
        return v
