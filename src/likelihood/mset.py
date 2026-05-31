"""
Construction, extraction, and reconstruction of Ncm.MSet instances.
"""

from enum import StrEnum
from typing import Callable, cast

from numcosmo_py import Nc, Ncm


class MassDefinition(StrEnum):
    """Mass definitions for halo mass parameters."""

    CRITICAL = "critical"
    MEAN = "mean"
    VIRIAL = "virial"

    def to_ncm(self) -> Nc.HaloMassSummaryMassDef:
        return _TO_NCM_MASS_DEF[self]

    @classmethod
    def from_ncm(cls, ncm_mass_def: Nc.HaloMassSummaryMassDef) -> "MassDefinition":
        return _FROM_NCM_MASS_DEF[ncm_mass_def]


_TO_NCM_MASS_DEF: dict[MassDefinition, Nc.HaloMassSummaryMassDef] = {
    MassDefinition.CRITICAL: Nc.HaloMassSummaryMassDef.CRITICAL,
    MassDefinition.MEAN: Nc.HaloMassSummaryMassDef.MEAN,
    MassDefinition.VIRIAL: Nc.HaloMassSummaryMassDef.VIRIAL,
}
_FROM_NCM_MASS_DEF = {v: k for k, v in _TO_NCM_MASS_DEF.items()}


class DensityProfile(StrEnum):
    """Density profiles for halo density profile parameters."""

    NFW = "nfw"

    def to_ncm(self) -> Callable[[Nc.HaloMassSummary], Nc.HaloDensityProfile]:
        return _TO_NCM_DENSITY_PROFILE[self]

    @classmethod
    def from_ncm(cls, profile: Nc.HaloDensityProfile) -> "DensityProfile":
        return _FROM_NCM_DENSITY_PROFILE[type(profile)]


_TO_NCM_DENSITY_PROFILE: dict[
    DensityProfile, Callable[[Nc.HaloMassSummary], Nc.HaloDensityProfile]
] = {
    DensityProfile.NFW: Nc.HaloDensityProfileNFW.new,
}
_FROM_NCM_DENSITY_PROFILE: dict[type[Nc.HaloDensityProfile], DensityProfile] = {
    Nc.HaloDensityProfileNFW: DensityProfile.NFW,
}


def create_mset(
    *,
    H0: float = 70.0,
    OmegaB: float = 0.045,
    OmegaC: float = 0.3 - 0.045,
    w: float = -1.0,
    Omegak: float = 0.0,
    mass_def: MassDefinition = MassDefinition.CRITICAL,
    mass_delta: float = 200.0,
    density_profile: DensityProfile = DensityProfile.NFW,
    ra: float = 0.0,
    dec: float = 0.0,
    z: float = 0.3,
    cDelta: float = 4.0,
    log10MDelta: float = 14.5,
) -> Ncm.MSet:
    """
    Create an Ncm.MSet containing the cosmology, halo mass summary, density profile,
    surface mass density, and halo position for a single lensing halo.

    Parameters
    ----------
    H0: float
        Hubble constant in km/s/Mpc.
    OmegaB: float
        Baryon density parameter.
    OmegaC: float
        Cold dark matter density parameter.
    w: float
        Dark energy equation of state parameter.
    Omegak: float
        Curvature density parameter.
    mass_def: MassDefinition
        Mass definition for the halo mass summary.
    mass_delta: float
        Overdensity for the halo mass summary.
    density_profile: DensityProfile
        Density profile for the halo density profile.
    ra: float
        Right ascension of the halo in degrees.
    dec: float
        Declination of the halo in degrees.
    z: float
        Redshift of the halo.
    cDelta: float
        Concentration parameter for the halo density profile.
    log10MDelta: float
        Log10 of the halo mass in units of solar masses for the halo mass summary.

    Returns
    -------
    Ncm.MSet
        An Ncm.MSet containing the cosmology, halo mass summary, density profile,
        surface mass density, and halo position for a single lensing halo.
    """
    cosmo = Nc.HICosmoDEXcdm()
    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()

    dist = Nc.Distance.new(6.0)
    hms = Nc.HaloCMParam.new(mass_def.to_ncm(), mass_delta)
    dp = density_profile.to_ncm()(hms)
    smd = Nc.WLSurfaceMassDensity.new(dist)
    hp = Nc.HaloPosition.new(dist)

    cosmo["H0"] = H0
    cosmo["Omegab"] = OmegaB
    cosmo["Omegac"] = OmegaC
    cosmo["w"] = w
    cosmo["Omegak"] = Omegak
    hp["ra"] = ra
    hp["dec"] = dec
    hp["z"] = z
    hms["cDelta"] = cDelta
    hms["log10MDelta"] = log10MDelta

    smd.prepare(cosmo)
    hp.prepare(cosmo)

    cosmo.param_set_desc("H0", {"fit": False})
    cosmo.param_set_desc("Omegac", {"fit": False})
    cosmo.param_set_desc("Omegab", {"fit": False})
    cosmo.param_set_desc("w", {"fit": False})
    cosmo.param_set_desc("Omegak", {"fit": False})
    hms.param_set_desc("cDelta", {"fit": False})
    hms.param_set_desc("log10MDelta", {"fit": True})

    return Ncm.MSet.new_array([cosmo, dp, smd, hp])


def extract_mset_params(mset: Ncm.MSet) -> dict:
    """
    Extract the parameters from an Ncm.MSet into a dict of simple types that can be
    pickled and used to reconstruct the mset with ``rebuild_mset``.

    Parameters
    ----------
    mset: Ncm.MSet
        The mset to extract parameters from.

    Returns
    -------
    dict
        A dict containing the parameters of the mset, with simple types that can be
        pickled.
    """
    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    hms = cast(Nc.HaloMassSummary, mset.peek_by_name("NcHaloMassSummary"))
    dp = cast(Nc.HaloDensityProfile, mset.peek_by_name("NcHaloDensityProfile"))

    return {
        "H0": cosmo["H0"],
        "OmegaB": cosmo["Omegab"],
        "OmegaC": cosmo["Omegac"],
        "w": cosmo["w"],
        "Omegak": cosmo["Omegak"],
        "mass_def": MassDefinition.from_ncm(hms.props.mass_def),
        "mass_delta": hms.props.Delta,
        "density_profile": DensityProfile.from_ncm(dp),
        "ra": hp["ra"],
        "dec": hp["dec"],
        "z": hp["z"],
        "cDelta": hms["cDelta"],
        "log10MDelta": hms["log10MDelta"],
    }


def rebuild_mset(params: dict) -> Ncm.MSet:
    """
    Rebuild an Ncm.MSet from a dict of parameters, as extracted by
    ``extract_mset_params``.

    Parameters
    ----------
    params: dict
        A dict containing the parameters of the mset, with simple types that can be
        pickled.

    Returns
    -------
    Ncm.MSet
        An Ncm.MSet containing the cosmology, halo mass summary, density profile,
        surface mass density, and halo position for a single lensing halo, reconstructed
        from the given parameters.
    """
    return create_mset(**params)
