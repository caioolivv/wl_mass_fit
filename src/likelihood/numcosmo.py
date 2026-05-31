from enum import StrEnum
from typing import cast, Tuple

import numpy as np
import pandas as pd
from numcosmo_py import Ncm, Nc
from numpy.typing import NDArray

from .likelihood import Likelihood

from pydantic import (
    Field,
    ConfigDict,
    PrivateAttr,
    field_validator,
    model_validator,
)
from scipy.optimize import fsolve


class ShapeModel(StrEnum):
    """Galaxy shape model expected by an NCLikelihood instance."""

    HSM_GAUSS = "hsm_gauss"
    HSM_GAUSS_GLOBAL = "hsm_gauss_global"


class NCLikelihood(Likelihood):
    """
    Likelihood class for the binned excess surface mass density profile described in
    Umetsu et al. (2020). Uses shapeHSM products.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    radius_bounds: Tuple[float, float] = Field(
        default=(0.4, 4.0),
        description=(
            "Minimum and maximum radii (in Mpc) to include in the likelihood."
        ),
    )
    shape_model: ShapeModel = Field(
        default=ShapeModel.HSM_GAUSS,
        description="Galaxy shape model: HSMGauss (per-galaxy std_shape) or HSMGaussGlobal (std_shape is a global free parameter).",
    )

    _data: Nc.GalaxyWLObs = PrivateAttr()

    @field_validator("radius_bounds")
    def validate_radius_bounds(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if any(r < 0 for r in v):
            raise ValueError("radius_bounds must be non-negative")
        if v[0] >= v[1]:
            raise ValueError("radius_bounds must be (min, max) with min < max")
        return v

    def _configure_models(self) -> None:
        """Fill in any galaxy SD submodels missing from ``self.mset``.

        Existing submodels are left untouched (so a mset loaded from a
        numcosmo experiment YAML keeps its tuned parameters). When the shape
        model is already on the mset, this validates it matches
        ``self.shape_model`` and otherwise raises.
        """
        mset = self.mset

        expected_shape_cls = (
            Nc.GalaxySDShapeHSMGauss
            if self.shape_model is ShapeModel.HSM_GAUSS
            else Nc.GalaxySDShapeHSMGaussGlobal
        )

        existing_shape = mset.peek_by_name("NcGalaxySDShape")
        if existing_shape is not None and not isinstance(
            existing_shape, expected_shape_cls
        ):
            raise ValueError(
                f"mset has NcGalaxySDShape of type "
                f"{type(existing_shape).__name__} but shape_model="
                f"{self.shape_model} expects {expected_shape_cls.__name__}"
            )

        needs_redshift = mset.peek_by_name("NcGalaxySDObsRedshift") is None
        needs_position = mset.peek_by_name("NcGalaxySDPosition") is None
        needs_shape = existing_shape is None

        if needs_position:
            hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
            cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
            ra = hp["ra"]
            dec = hp["dec"]

            hp.prepare(cosmo)

            min_dec = dec - abs(
                fsolve(
                    lambda sep: hp.projected_radius_from_ra_dec(
                        cosmo, float(ra), float(dec) - float(sep[0])
                    )
                    - float(10 / cosmo.h()),
                    0.5,
                )[0]
            )
            max_dec = dec + abs(
                fsolve(
                    lambda sep: hp.projected_radius_from_ra_dec(
                        cosmo, float(ra), float(dec) + float(sep[0])
                    )
                    - float(10 / cosmo.h()),
                    0.5,
                )[0]
            )
            min_ra = ra - abs(
                fsolve(
                    lambda sep: hp.projected_radius_from_ra_dec(
                        cosmo, float(ra) - float(sep[0]), float(dec)
                    )
                    - float(10 / cosmo.h()),
                    0.5,
                )[0]
            )
            max_ra = ra + abs(
                fsolve(
                    lambda sep: hp.projected_radius_from_ra_dec(
                        cosmo, float(ra) + float(sep[0]), float(dec)
                    )
                    - float(10 / cosmo.h()),
                    0.5,
                )[0]
            )
            mset.set(
                Nc.GalaxySDPositionFlat.new(
                    float(min_ra), float(max_ra), float(min_dec), float(max_dec)
                )
            )

        if needs_redshift:
            mset.set(Nc.GalaxySDObsRedshiftPz.new())

        if needs_shape:
            mset.set(expected_shape_cls.new(Nc.GalaxyWLObsEllipConv.TRACE))

    @model_validator(mode="after")
    def validate_model(self):
        self._configure_models()
        return self

    def prepare_data(self) -> None:
        """Compute the data vector and related quantities from the observation data."""
        data_cut = self.obs.copy()
        data_cut = data_cut[
            (data_cut["radius"] >= self.radius_bounds[0])
            & (data_cut["radius"] <= self.radius_bounds[1])
        ].reset_index(drop=True)

        sd_redshift = cast(
            Nc.GalaxySDObsRedshiftPz, self.mset.peek_by_name("NcGalaxySDObsRedshift")
        )
        sd_position = cast(
            Nc.GalaxySDPositionFlat, self.mset.peek_by_name("NcGalaxySDPosition")
        )
        sd_shape = cast(Nc.GalaxySDShape, self.mset.peek_by_name("NcGalaxySDShape"))

        z_data = Nc.GalaxySDObsRedshiftData.new(sd_redshift)
        p_data = Nc.GalaxySDPositionData.new(sd_position, z_data)
        s_data = Nc.GalaxySDShapeData.new(sd_shape, p_data)

        obs_dict: dict[str, list] = {
            "ra": data_cut["i_ra"].values.tolist(),
            "dec": data_cut["i_dec"].values.tolist(),
            "epsilon_obs_1": data_cut["i_hsmshaperegauss_e1"].values.tolist(),
            "epsilon_obs_2": data_cut["i_hsmshaperegauss_e2"].values.tolist(),
            "std_noise": data_cut["i_hsmshaperegauss_derived_sigma_e"].values.tolist(),
            "m": data_cut["i_hsmshaperegauss_derived_shear_bias_m"].values.tolist(),
            "c1": data_cut["i_hsmshaperegauss_derived_shear_bias_c1"].values.tolist(),
            "c2": data_cut["i_hsmshaperegauss_derived_shear_bias_c2"].values.tolist(),
            "z": data_cut["z"].values.tolist(),
            "pz": data_cut["pz_spline"].values.tolist(),
        }
        if self.shape_model is ShapeModel.HSM_GAUSS:
            obs_dict["std_shape"] = data_cut[
                "i_hsmshaperegauss_derived_rms_e"
            ].values.tolist()

        self._data = Nc.GalaxyWLObs.new(
            Nc.GalaxyWLObsEllipConv.TRACE,
            self.coord_system.to_ncm(),
            len(data_cut),
            s_data.required_columns(),
        )

        for i in range(len(obs_dict["ra"])):
            for key, value in obs_dict.items():
                if key == "pz":
                    self._data.set_pz(i, value[i])
                else:
                    self._data.set(key, i, value[i])

    def maximum_likelihood_estimate(self) -> NDArray[np.float64]:
        """Return best-fit [log10M] via scipy.optimize.minimize."""
        data_cluster = Nc.DataClusterWL.new()
        data_cluster.set_obs(self._data)
        data_cluster.set_init(True)
        data_cluster.set_integ_method(Nc.DataClusterWLIntegMethod.FIXED_NODES)

        self.mset.prepare_fparam_map()

        dataset = Ncm.Dataset.new_array([data_cluster])
        likelihood = Ncm.Likelihood.new(dataset)
        fit = Ncm.Fit.factory(
            Ncm.FitType.NLOPT,
            None,
            likelihood,
            self.mset,
            Ncm.FitGradType.NUMDIFF_FORWARD,
        )
        fit.run(Ncm.FitRunMsgs.SIMPLE)

        results = []
        for param in self.fparams:
            model_name, param_name = param.split(":", 1)
            results.append(self.mset.peek_by_name(model_name)[param_name])

        return np.array(results)

    def sample_posterior(
        self,
        nsamples: int,
        nwalkers: int = 200,
        nthreads: int = 4,
        progress: bool = True,
        filename: str = "numcosmo_mcmc.fits",
    ) -> pd.DataFrame:
        """Run emcee with per-process mset reconstruction."""
        data_cluster = Nc.DataClusterWL.new()
        data_cluster.set_obs(self._data)
        data_cluster.set_init(True)
        data_cluster.set_integ_method(Nc.DataClusterWLIntegMethod.FIXED_NODES)

        self.mset.prepare_fparam_map()

        dataset = Ncm.Dataset.new_array([data_cluster])
        likelihood = Ncm.Likelihood.new(dataset)
        fit = Ncm.Fit.factory(
            Ncm.FitType.NLOPT,
            None,
            likelihood,
            self.mset,
            Ncm.FitGradType.NUMDIFF_FORWARD,
        )

        apes_walker = Ncm.FitESMCMCWalkerAPES.new(nwalkers, len(self.fparams))

        apes_walker.set_use_threads(True)

        init_sampler = Ncm.MSetTransKernGauss.new(0)
        init_sampler.set_mset(self.mset)
        init_sampler.set_prior_from_mset()
        init_sampler.set_cov_from_rescale(1.0)

        esmcmc = Ncm.FitESMCMC.new(
            fit,
            nwalkers,
            init_sampler,
            apes_walker,
            Ncm.FitRunMsgs.SIMPLE,
        )
        esmcmc.set_nthreads(nthreads)
        esmcmc.set_data_file(filename)

        esmcmc.start_run()
        esmcmc.run(nsamples)
        esmcmc.end_run()

        mcat = esmcmc.get_catalog()
        m2lnL = mcat.get_m2lnp_var()
        rows = np.array([mcat.peek_row(i).dup_array() for i in range(mcat.len())])

        # Extract fitted-parameter columns in self.fparams order.
        # The catalog columns are in fparam-map order, which may differ from self.fparams order.
        short_names = [p.split(":", 1)[1] for p in self.fparams]

        # Build list of (fparam_index_in_mset, column_in_mcat) for each param in self.fparams
        self.mset.prepare_fparam_map()
        param_cols = []
        for param in self.fparams:
            model_name, param_name = param.split(":", 1)
            for i in range(self.mset.fparams_len()):
                pi = self.mset.fparam_get_pi(i)
                model = self.mset.peek(pi.mid)
                if (model_name == model.describe().name and
                    param_name == model.param_name(pi.pid)):
                    param_cols.append(i)
                    break

        samples = pd.DataFrame(rows[:, param_cols], columns=short_names)

        return samples
