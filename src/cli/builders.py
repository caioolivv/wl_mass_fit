"""
Shared helpers that construct library objects from CLI flag values.
"""

from enum import StrEnum
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
from numcosmo_py import Ncm

from ..likelihood.likelihood import Likelihood
from ..likelihood.binned_shear import BinnedShearLikelihood
from ..likelihood.binned_sigma import BinnedSigmaLikelihood
from ..likelihood.wtg import WtGLikelihood
from ..likelihood.mset import create_mset, MassDefinition, DensityProfile
from ..mock.generator import MockGenerator
from ..mock.mset import create_mock_mset
from ..mock.source import DataFrameGalaxySource, HamanaGalaxySource
from ..utils.utils import CoordSystem


class LikelihoodType(StrEnum):
    BINNED_SHEAR = "binned-shear"
    BINNED_SIGMA = "binned-sigma"
    WTG = "wtg"


def build_mset(
    *,
    H0: float = 70.0,
    OmegaB: float = 0.045,
    OmegaC: float = 0.255,
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
    return create_mset(
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


def build_mock_mset(
    *,
    H0: float = 70.0,
    OmegaB: float = 0.045,
    OmegaC: float = 0.255,
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
    return create_mock_mset(
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


def build_generator(
    mset: Ncm.MSet,
    *,
    clusters_path: Path,
    true_ra: float,
    true_dec: float,
    r_min: float,
    r_max: float,
    r_miscenter: float,
    n_gals: int,
    p_cut: float = 0.98,
    delta_z: float = 0.2,
    z_max: float = 2.5,
    coord_system: CoordSystem = CoordSystem.CELESTIAL,
) -> MockGenerator:
    source = HamanaGalaxySource.from_directory(clusters_path)
    return MockGenerator(
        mset=mset,
        source=source,
        true_ra=true_ra,
        true_dec=true_dec,
        r_min=r_min,
        r_max=r_max,
        r_miscenter=r_miscenter,
        n_gals=n_gals,
        p_cut=p_cut,
        delta_z=delta_z,
        z_max=z_max,
        coord_system=coord_system,
    )


def build_likelihood_factory(
    fit_mset: Ncm.MSet,
    *,
    likelihood_type: LikelihoodType,
    coord_system: CoordSystem,
    fparams: Sequence[str],
    fparams_bounds: Optional[Sequence[tuple[float, float]]],
    bin_edges: Optional[Sequence[float]],
    radius_bounds: Optional[tuple[float, float]],
) -> Callable[[pd.DataFrame], Likelihood]:
    """Return a callable that builds the chosen likelihood for a given obs DataFrame."""

    # Save initial fparam values so each factory call starts from the same point.
    _initial: list[tuple[str, str, float]] = []
    for full_name in fparams:
        model_name, param_name = full_name.split(":", 1)
        model = fit_mset.peek_by_name(model_name)
        if model is not None:
            _initial.append((model_name, param_name, model[param_name]))

    def factory(obs: pd.DataFrame) -> Likelihood:
        # Reset fit_mset to initial param values before each likelihood construction.
        for model_name, param_name, value in _initial:
            fit_mset.peek_by_name(model_name)[param_name] = value

        kwargs = dict(
            mset=fit_mset,
            obs=obs,
            coord_system=coord_system,
            fparams=fparams,
            fparams_bounds=fparams_bounds,
        )
        if likelihood_type == LikelihoodType.BINNED_SHEAR:
            if bin_edges is not None:
                kwargs["bin_edges"] = bin_edges
            return BinnedShearLikelihood(**kwargs)
        elif likelihood_type == LikelihoodType.BINNED_SIGMA:
            if bin_edges is not None:
                kwargs["bin_edges"] = bin_edges
            return BinnedSigmaLikelihood(**kwargs)
        elif likelihood_type == LikelihoodType.WTG:
            if radius_bounds is not None:
                kwargs["radius_bounds"] = radius_bounds
            return WtGLikelihood(**kwargs)
        else:
            raise ValueError(f"Unknown likelihood type: {likelihood_type}")

    return factory


def parse_fparams(fparams_str: Optional[str]) -> list[str]:
    """Parse comma-separated 'Model:param' strings."""
    if not fparams_str:
        return ["NcHaloMassSummary:log10MDelta"]
    return [s.strip() for s in fparams_str.split(",")]


def parse_fparams_bounds(bounds_str: Optional[str]) -> Optional[list[tuple[float, float]]]:
    """Parse 'lo1:hi1,lo2:hi2' into list of (lo, hi) tuples."""
    if not bounds_str:
        return None
    pairs = []
    for pair in bounds_str.split(","):
        lo, hi = pair.strip().split(":")
        pairs.append((float(lo), float(hi)))
    return pairs


def parse_bin_edges(edges_str: Optional[str]) -> Optional[list[float]]:
    """Parse comma-separated float string into list."""
    if not edges_str:
        return None
    return [float(e.strip()) for e in edges_str.split(",")]


def parse_radius_bounds(bounds_str: Optional[str]) -> Optional[tuple[float, float]]:
    """Parse 'lo,hi' string into (lo, hi) tuple."""
    if not bounds_str:
        return None
    lo, hi = bounds_str.strip().split(",")
    return (float(lo), float(hi))
