"""
Likelihood module for computing likelihoods of data given a model.
"""

from ..utils.utils import (
    compute_radius,
    compute_cross_component,
    compute_tangential_component,
    CoordSystem,
    create_ncm_spline,
)
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    ValidationInfo,
)
from typing import ClassVar, Sequence, Set, Tuple
from numcosmo_py import Nc, Ncm
import numpy as np
from numpy.typing import NDArray
import pandas as pd

default_cosmo = Nc.HICosmoDEXcdm()

default_cosmo.omega_x2omega_k()

default_cosmo["H0"] = 70
default_cosmo["Omegab"] = 0.045
default_cosmo["Omegac"] = 0.3 - 0.045
default_cosmo["w"] = -1.0
default_cosmo["Omegak"] = 0.00
default_dist = Nc.Distance.new(6.0)
default_hms = Nc.HaloCMParam.new(Nc.HaloMassSummaryMassDef.CRITICAL, 500.0)
default_dp = Nc.HaloDensityProfileNFW.new(default_hms)
default_smd = Nc.WLSurfaceMassDensity.new(default_dist)
default_hp = Nc.HaloPosition.new(default_dist)

default_cosmo.param_set_desc("H0", {"fit": False})
default_cosmo.param_set_desc("Omegac", {"fit": False})
default_cosmo.param_set_desc("Omegab", {"fit": False})
default_cosmo.param_set_desc("w", {"fit": False})
default_cosmo.param_set_desc("Omegak", {"fit": False})
default_hms.param_set_desc("cDelta", {"fit": False})
default_hms.param_set_desc("log10MDelta", {"fit": True})

default_mset = Ncm.MSet.new_array([default_cosmo, default_dp, default_smd, default_hp])


class Likelihood(BaseModel, ABC):
    """
    A data structure to hold the likelihood information, including the model set and
    the data, based on Pandas DataFrames.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        validate_assignment=True,
    )

    mset: Ncm.MSet = Field(
        default=default_mset, description="Model set for likelihood computation."
    )
    fparams: Sequence[str] = Field(
        default=["NcHaloMassSummary:log10MDelta"],
        description="Names of free parameters for MCMC sampling.",
    )
    param_bounds: Sequence[Tuple[float, float]] | None = Field(
        default=None, description="Parameter bounds for MCMC sampling."
    )
    obs: pd.DataFrame = Field(
        ..., description="Observables for likelihood computation."
    )
    coord_system: CoordSystem = Field(
        default=CoordSystem.CELESTIAL,
        description=(
            "Coordinate system of the input data (e.g., 'celestial' or 'cartesian')."
        ),
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

    @field_validator("mset")
    @classmethod
    def validate_mset(cls, v: Ncm.MSet) -> Ncm.MSet:
        if not isinstance(v, Ncm.MSet):
            raise ValueError("mset must be a NcmMSet.")

        if v.peek_by_name("NcHICosmo") is None:
            raise ValueError("mset must contain a NcHIcosmo submodel.")

        if v.peek_by_name("NcHaloDensityProfile") is None:
            raise ValueError("mset must contain a NcHaloDensityProfile submodel.")

        if v.peek_by_name("NcWLSurfaceMassDensity") is None:
            raise ValueError("mset must contain a NcWLSurfaceMassDensity submodel.")

        if v.peek_by_name("NcHaloPosition") is None:
            raise ValueError("mset must contain a NcHaloPosition submodel.")

        if v.peek_by_name("NcHaloMassSummary") is None:
            raise ValueError("mset must contain a NcHaloMassSummary submodel.")

        return v

    @field_validator("fparams")
    @classmethod
    def validate_fparams(cls, v: Sequence[str], info: ValidationInfo) -> Sequence[str]:
        mset = info.data.get("mset", default_mset)

        for param in v:
            if mset.param_get_by_full_name(param) is None:
                raise ValueError(f"Parameter '{param}' not found in mset.")

        for i in range(mset.fparams_len()):
            mid = mset.fparam_get_mid(i)
            pid = mset.fparam_get_pid(i)
            model = mset.peek(mid)
            model.param_set_desc(model.param_name(pid), {"fit": False})

        for param in v:
            model_name, param_name = param.split(":", 1)
            mset.peek_by_name(model_name).param_set_desc(param_name, {"fit": True})

        return v

    @field_validator("param_bounds")
    @classmethod
    def validate_param_bounds(
        cls, v: Sequence[Tuple[float, float]] | None, info: ValidationInfo
    ) -> Sequence[Tuple[float, float]] | None:
        fparams = info.data.get("fparams", ["NcHaloMassSummary:log10MDelta"])
        mset = info.data.get("mset", default_mset)

        if v is None:
            return v

        if len(v) != len(fparams):
            raise ValueError("Length of param_bounds must match length of fparams.")

        for param, bounds in zip(fparams, v):
            model_name, param_name = param.split(":", 1)
            mset.peek_by_name(model_name).param_set_desc(
                param_name, {"lower-bound": bounds[0], "upper-bound": bounds[1]}
            )

        return v

    @field_validator("obs")
    @classmethod
    def validate_obs(cls, v: pd.DataFrame, info: ValidationInfo) -> pd.DataFrame:
        if not isinstance(v, pd.DataFrame):
            raise ValueError("obs must be a pandas DataFrame.")

        if not cls.REQUIRED_COLUMNS.issubset(v.columns):
            missing_cols = cls.REQUIRED_COLUMNS - set(v.columns)

            raise ValueError(f"Missing required columns: {missing_cols}")

        mset = info.data.get("mset", default_mset)
        coord_system = info.data.get("coord_system", CoordSystem.CELESTIAL)
        v["pz_spline"] = v.apply(
            lambda row: create_ncm_spline(row["pz_nodes"], row["pz_weights"]), axis=1
        )
        v["radius"] = v.apply(
            lambda row: compute_radius(row["i_ra"], row["i_dec"], mset), axis=1
        )
        v["e_t"] = v.apply(
            lambda row: compute_tangential_component(
                row["i_hsmshaperegauss_e1"],
                row["i_hsmshaperegauss_e2"],
                row["i_ra"],
                row["i_dec"],
                coord_system,
                mset,
            ),
            axis=1,
        )
        v["e_x"] = v.apply(
            lambda row: compute_cross_component(
                row["i_hsmshaperegauss_e1"],
                row["i_hsmshaperegauss_e2"],
                row["i_ra"],
                row["i_dec"],
                coord_system,
                mset,
            ),
            axis=1,
        )

        return v

    @abstractmethod
    def maximum_likelihood_estimate(self) -> NDArray[np.float64]:
        """
        Compute the maximum likelihood estimate of the model parameters given the data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def sample_posterior(
        self,
        nsamples: int,
        nwalkers: int,
        nthreads: int,
        progress: bool,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """
        Sample the posterior distribution of the model parameters given the data.

        Parameters
        ----------
        nwalkers : float
            The number of MCMC walkers to use.
        nsamples : float
            The number of MCMC samples to draw.
        nthreads : float
            The number of threads to use for parallel sampling.
        progress : bool
            Whether to display a progress bar during sampling.
        filename : str | None
            Optional filename to save the MCMC samples as a CSV file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
