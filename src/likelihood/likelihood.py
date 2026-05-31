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
from .mset import create_mset
from abc import ABC, abstractmethod
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    ValidationInfo,
    PrivateAttr,
)
from typing import cast, ClassVar, Sequence, Set, Tuple
from numcosmo_py import Ncm, Nc
import numpy as np
from numpy.typing import NDArray
import pandas as pd


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
        default_factory=create_mset,
        description="Model set for likelihood computation.",
    )
    fparams: Sequence[str] = Field(
        default=["NcHaloMassSummary:log10MDelta"],
        description="Names of free parameters for MCMC sampling.",
    )
    fparams_bounds: Sequence[Tuple[float, float]] | None = Field(
        default=None, description="Parameters bounds for MCMC sampling."
    )
    coord_system: CoordSystem = Field(
        description=(
            "Coordinate system of the input data (e.g., 'celestial' or 'cartesian')."
        ),
    )
    obs: pd.DataFrame = Field(
        ..., description="Observables for likelihood computation."
    )

    _posterior: pd.DataFrame = PrivateAttr()

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
        "z",
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
    def validate_fparams(cls, v: Sequence[str]) -> Sequence[str]:
        if not isinstance(v, Sequence) or isinstance(v, str):
            raise ValueError("fparams must be a sequence of strings.")

        for param in v:
            if not isinstance(param, str) or ":" not in param:
                raise ValueError(
                    "Each fparam must be a string in the format 'ModelName:ParamName'."
                )

        return v

    def _configure_fparams(self) -> None:
        """Apply fitted-parameter flags and bounds to the instance mset."""
        mset = self.mset

        mset.prepare_fparam_map()

        for param in self.fparams:
            if mset.param_get_by_full_name(param) is None:
                raise ValueError(f"Parameter '{param}' not found in mset.")

        for i in range(mset.fparams_len()):
            mid = mset.fparam_get_pi(i).mid
            pid = mset.fparam_get_pi(i).pid
            model = mset.peek(mid)
            model.param_set_desc(model.param_name(pid), {"fit": False})

        for param in self.fparams:
            model_name, param_name = param.split(":", 1)
            mset.peek_by_name(model_name).param_set_desc(param_name, {"fit": True})

        if self.fparams_bounds is None:
            return

        if len(self.fparams_bounds) != len(self.fparams):
            raise ValueError("Length of param_bounds must match length of fparams.")

        for param, bounds in zip(self.fparams, self.fparams_bounds):
            model_name, param_name = param.split(":", 1)
            mset.peek_by_name(model_name).param_set_desc(
                param_name, {"lower-bound": bounds[0], "upper-bound": bounds[1]}
            )

    def __init__(self, **data):
        """Initialize and configure fparams on the mset."""
        super().__init__(**data)
        self._configure_fparams()

    @field_validator("obs")
    @classmethod
    def validate_obs(cls, v: pd.DataFrame, info: ValidationInfo) -> pd.DataFrame:
        if not isinstance(v, pd.DataFrame):
            raise ValueError("obs must be a pandas DataFrame.")

        if not cls.REQUIRED_COLUMNS.issubset(v.columns):
            missing_cols = cls.REQUIRED_COLUMNS - set(v.columns)

            raise ValueError(f"Missing required columns: {missing_cols}")

        mset = info.data.get("mset", create_mset())
        coord_system = info.data.get("coord_system", CoordSystem.CELESTIAL)

        # Reuse pre-built splines from the generator when present
        if "pz_spline" not in v.columns:
            v["pz_spline"] = [
                create_ncm_spline(w, n)
                for w, n in zip(v["pz_weights"], v["pz_nodes"])
            ]

        # Prepare once; compute radius and polar angle in a single pass per galaxy
        hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
        cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
        hp.prepare(cosmo)

        ras = v["i_ra"].to_numpy()
        decs = v["i_dec"].to_numpy()

        v["radius"] = [hp.projected_radius_from_ra_dec(cosmo, ra, dec) for ra, dec in zip(ras, decs)]

        phis = np.array([hp.polar_angles(ra, dec)[1] for ra, dec in zip(ras, decs)])
        if coord_system == CoordSystem.EUCLIDEAN:
            phis = np.pi - phis

        e_complex = v["i_hsmshaperegauss_e1"].to_numpy() + 1j * v["i_hsmshaperegauss_e2"].to_numpy()
        rotated = e_complex * np.exp(-2j * phis)
        v["e_t"] = np.real(rotated)
        v["e_x"] = np.imag(rotated)

        return v

    @abstractmethod
    def prepare_data(self) -> None:
        """
        Prepare the data for likelihood computation, such as computing derived
        quantities or precomputing model predictions.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
        filename: str,
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
