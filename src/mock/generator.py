"""
Mock galaxy cluster observation generator.
"""

from typing import cast
from scipy.optimize import fsolve
from pydantic import BaseModel, ConfigDict, Field, field_validator

from numcosmo_py import Ncm, Nc
import numpy as np
import pandas as pd

from ..utils.utils import CoordSystem, create_ncm_spline
from .source import GalaxySource


class MockGenerator(BaseModel):
    """
    Generate mock galaxy cluster weak-lensing observations from real galaxy sources.

    Produces a pandas DataFrame with the 12 columns required by Likelihood.obs:
    i_ra, i_dec, i_hsmshaperegauss_e1, i_hsmshaperegauss_e2,
    i_hsmshaperegauss_derived_rms_e, i_hsmshaperegauss_derived_sigma_e,
    i_hsmshaperegauss_derived_shear_bias_m, i_hsmshaperegauss_derived_shear_bias_c1,
    i_hsmshaperegauss_derived_shear_bias_c2, z, pz_weights, pz_nodes.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    mset: Ncm.MSet = Field(
        description="Model set containing cosmology and galaxy distribution submodels."
    )
    source: GalaxySource = Field(description="Galaxy property source for sampling.")
    true_ra: float = Field(description="True RA of halo center in degrees.")
    true_dec: float = Field(description="True Dec of halo center in degrees.")
    z: float = Field(description="True redshift of halo center.")
    r_min: float = Field(description="Minimum sampling radius in Mpc/h.")
    r_max: float = Field(description="Maximum sampling radius in Mpc/h.")
    r_miscenter: float = Field(description="Miscentering offset in Mpc/h.")
    n_gals: int = Field(description="Number of galaxies to generate.")
    p_cut: float = Field(
        default=0.98,
        description="Photo-z cumulative probability cut.",
    )
    delta_z: float = Field(
        default=0.2,
        description="Photo-z catastrophic redshift threshold (z_cut = z_cluster + delta_z).",
    )
    z_max: float = Field(
        default=2.5,
        description="Maximum individual galaxy redshift.",
    )
    coord_system: CoordSystem = Field(
        default=CoordSystem.CELESTIAL,
        description="Coordinate system (CELESTIAL or EUCLIDEAN).",
    )

    @field_validator("mset")
    @classmethod
    def validate_mset(cls, v: Ncm.MSet) -> Ncm.MSet:
        """Ensure mset has the required likelihood submodels."""
        required = [
            "NcHICosmo",
            "NcHaloDensityProfile",
            "NcWLSurfaceMassDensity",
            "NcHaloPosition",
            "NcHaloMassSummary",
        ]
        for name in required:
            if v.peek_by_name(name) is None:
                raise ValueError(f"mset must contain {name} submodel.")
        return v

    @field_validator("n_gals")
    @classmethod
    def validate_n_gals(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_gals must be positive.")
        return v

    @field_validator("r_min", "r_max", "r_miscenter")
    @classmethod
    def validate_radii(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Radii must be non-negative.")
        return v

    def generate(self, rng: Ncm.RNG) -> pd.DataFrame:
        """
        Generate a mock cluster observation.

        Parameters
        ----------
        rng : Ncm.RNG
            NumCosmo random number generator.

        Returns
        -------
        pd.DataFrame
            Raw DataFrame with the 12 required columns and per-row P(z) arrays.
        """
        cosmo = cast(Nc.HICosmo, self.mset.peek_by_name("NcHICosmo"))
        halo_position = cast(Nc.HaloPosition, self.mset.peek_by_name("NcHaloPosition"))
        halo_position["ra"] = self.true_ra
        halo_position["dec"] = self.true_dec
        halo_position["z"] = self.z
        halo_position.prepare(cosmo)

        # Convert radii from Mpc/h to Mpc
        h = cosmo.h()
        r_min_mpc = self.r_min / h
        r_max_mpc = self.r_max / h
        r_miscenter_mpc = self.r_miscenter / h

        # Compute miscentering offset
        if r_miscenter_mpc > 0.0:

            def residual(sep):
                r = halo_position.projected_radius_from_ra_dec(
                    cosmo, self.true_ra + sep[0], self.true_dec
                )
                return r - r_miscenter_mpc

            delta_ra = abs(float(fsolve(residual, [0.01])[0]))
        else:
            delta_ra = 0.0

        ra_assumed = self.true_ra + delta_ra
        dec_assumed = self.true_dec

        # Compute RA/Dec bounds enclosing the r_max circle around the assumed position
        halo_position["ra"] = ra_assumed
        halo_position["dec"] = dec_assumed
        halo_position.prepare(cosmo)

        def _ra_residual(delta):
            return (
                halo_position.projected_radius_from_ra_dec(
                    cosmo, ra_assumed + delta[0], dec_assumed
                )
                - r_max_mpc
            )

        def _dec_residual(delta):
            return (
                halo_position.projected_radius_from_ra_dec(
                    cosmo, ra_assumed, dec_assumed + delta[0]
                )
                - r_max_mpc
            )

        delta_ra_extent = abs(float(fsolve(_ra_residual, [0.1])[0]))
        delta_dec_extent = abs(float(fsolve(_dec_residual, [0.1])[0]))

        # Create galaxy distribution submodels locally
        sd_redshift = Nc.GalaxySDObsRedshiftPz.new()
        sd_shape = Nc.GalaxySDShapeHSMGauss.new(Nc.GalaxyWLObsEllipConv.TRACE)
        sd_position = Nc.GalaxySDPositionFlat.new(
            ra_assumed - delta_ra_extent,
            ra_assumed + delta_ra_extent,
            dec_assumed - delta_dec_extent,
            dec_assumed + delta_dec_extent,
        )

        # Prepare NumCosmo data structures
        z_data = Nc.GalaxySDObsRedshiftData.new(sd_redshift)
        p_data = Nc.GalaxySDPositionData.new(sd_position, z_data)
        s_data = Nc.GalaxySDShapeData.new(sd_shape, p_data)

        rows = []
        z_halo = halo_position["z"]
        rng_np = np.random.default_rng()  # For galaxy source sampling

        for gal_idx in range(self.n_gals):
            # Photo-z rejection loop (max 100 attempts)
            pz_attempts = 0
            while pz_attempts < 100:
                pz_attempts += 1
                props = self.source.sample(z_halo, rng_np)
                gal_z = props.z

                # Reuse pre-built spline from source when available
                if props.pz_spline is not None:
                    pz_spline = props.pz_spline
                else:
                    pz_spline = create_ncm_spline(props.pz_weights, props.pz_nodes)
                    pz_spline.prepare()

                pz_zmin, pz_zmax = pz_spline.get_bounds()
                z_min = z_halo + self.delta_z

                if z_min >= pz_zmax:
                    continue
                if gal_z > self.z_max:
                    continue
                if pz_spline.eval_integ(max(pz_zmin, z_min), pz_zmax) < self.p_cut:
                    continue

                z_data.z = gal_z
                sd_redshift.data_set(z_data, pz_spline)
                break

            if pz_attempts >= 100:
                raise RuntimeError(
                    f"Photo-z rejection failed after 100 attempts for galaxy {gal_idx}"
                )

            # Position sampling loop (max 100 attempts)
            halo_position["ra"] = ra_assumed
            halo_position["dec"] = dec_assumed
            halo_position.prepare(cosmo)

            pos_attempts = 0
            while pos_attempts < 100:
                pos_attempts += 1
                sd_position.gen(self.mset, p_data, rng)

                ra_gal = p_data.ra
                dec_gal = p_data.dec
                r_assumed = float(
                    halo_position.projected_radius_from_ra_dec(cosmo, ra_gal, dec_gal)
                )

                if r_min_mpc < r_assumed < r_max_mpc:
                    break

            if pos_attempts >= 100:
                raise RuntimeError(
                    f"Position sampling failed after 100 attempts for galaxy {gal_idx}"
                )

            # Shape generation (true center)
            halo_position["ra"] = self.true_ra
            halo_position["dec"] = self.true_dec
            halo_position.prepare(cosmo)

            sd_shape.gen(
                self.mset,
                s_data,
                props.rms_e,    # std_shape: intrinsic shape dispersion
                props.sigma_e,  # std_noise: per-galaxy measurement error
                props.c1,
                props.c2,
                props.m,
                self.coord_system.to_ncm(),
                rng,
            )

            # Record the OBSERVED (lensed) ellipticity, not the intrinsic one.
            # gen() stores the sheared + biased + noised shape in epsilon_obs;
            # epsilon_int holds the unlensed source shape (carries no mass signal).
            epsilon_obs_1, epsilon_obs_2, *_ = sd_shape.data_get(s_data)

            # Record row (observed ellipticity + per-galaxy properties)
            row = {
                "i_ra": p_data.ra,
                "i_dec": p_data.dec,
                "i_hsmshaperegauss_e1": epsilon_obs_1,
                "i_hsmshaperegauss_e2": epsilon_obs_2,
                "i_hsmshaperegauss_derived_sigma_e": props.sigma_e,
                "i_hsmshaperegauss_derived_rms_e": props.rms_e,
                "i_hsmshaperegauss_derived_shear_bias_m": props.m,
                "i_hsmshaperegauss_derived_shear_bias_c1": props.c1,
                "i_hsmshaperegauss_derived_shear_bias_c2": props.c2,
                "z": props.z,
                "pz_nodes": props.pz_nodes,
                "pz_weights": props.pz_weights,
                "pz_spline": pz_spline,
            }
            rows.append(row)

        return pd.DataFrame(rows)
