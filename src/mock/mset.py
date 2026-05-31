"""
Helper for creating mock mset with proper parameters for mock generation.
"""

from ..likelihood.mset import create_mset as create_base_mset
from ..likelihood.mset import MassDefinition, DensityProfile


def create_mock_mset(
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
) -> "Ncm.MSet":
    """
    Create an Ncm.MSet suitable for mock generation.

    This returns the base likelihood mset with cosmology and halo models.
    The MockGenerator adds galaxy distribution submodels locally.

    Parameters follow create_mset from src/likelihood/mset.py.
    """
    return create_base_mset(
        H0=H0,
        OmegaB=OmegaB,
        OmegaC=OmegaC,
        w=w,
        Omegak=Omegak,
        mass_def=mass_def,
        mass_delta=mass_delta,
        density_profile=density_profile,
        ra=ra,
        dec=dec,
        z=z,
        cDelta=cDelta,
        log10MDelta=log10MDelta,
    )
