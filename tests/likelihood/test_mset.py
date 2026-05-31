"""
Test suite for the mset module.
"""

from src.likelihood.mset import (
    create_mset,
    extract_mset_params,
    rebuild_mset,
    MassDefinition,
    DensityProfile,
)
from typing import cast
import pytest
import numpy as np
from numcosmo_py import Ncm, Nc


@pytest.fixture(
    name="mass_def",
    params=[MassDefinition.CRITICAL, MassDefinition.MEAN, MassDefinition.VIRIAL],
)
def fixture_mass_def(request) -> MassDefinition:
    """Fixture providing different MassDefinition values for testing."""
    return request.param


@pytest.fixture(name="density_profile", params=[DensityProfile.NFW])
def fixture_density_profile(request) -> DensityProfile:
    """Fixture providing different DensityProfile values for testing."""
    return request.param


def test_create_mset_defaults():
    """Test that create_mset with default args produces the documented values."""
    mset = create_mset()
    assert isinstance(mset, Ncm.MSet)

    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    assert cosmo is not None
    assert cosmo["H0"] == 70.0
    assert cosmo["Omegab"] == 0.045
    assert cosmo["Omegac"] == 0.255
    assert cosmo["w"] == -1.0
    assert cosmo["Omegak"] == 0.0

    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    assert hp is not None
    assert hp["ra"] == 0.0
    assert hp["dec"] == 0.0
    assert hp["z"] == 0.3

    hms = cast(Nc.HaloMassSummary, mset.peek_by_name("NcHaloMassSummary"))
    assert hms is not None
    assert hms["cDelta"] == 4.0
    assert hms["log10MDelta"] == 14.5


def test_create_mset_random(mass_def, density_profile):
    """Test that create_mset propagates explicit kwargs into the mset."""
    for _ in range(5):
        H0 = np.random.uniform(60, 80)
        OmegaB = np.random.uniform(0.03, 0.06)
        OmegaC = np.random.uniform(0.2, 0.4)
        w = np.random.uniform(-1.5, -0.5)
        Omegak = np.random.uniform(-0.05, 0.05)
        mass_delta = np.random.uniform(100, 500)
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-90, 90)
        z = np.random.uniform(0.1, 0.5)
        cDelta = np.random.uniform(3, 6)
        log10MDelta = np.random.uniform(13, 15)

        mset = create_mset(
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
        assert isinstance(mset, Ncm.MSet)

        cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
        assert cosmo is not None
        assert cosmo["H0"] == H0
        assert cosmo["Omegab"] == OmegaB
        assert cosmo["Omegac"] == OmegaC
        assert cosmo["w"] == w
        assert cosmo["Omegak"] == Omegak

        hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
        assert hp is not None
        assert hp["ra"] == ra
        assert hp["dec"] == dec
        assert hp["z"] == z

        hms = cast(Nc.HaloMassSummary, mset.peek_by_name("NcHaloMassSummary"))
        assert hms is not None
        assert hms["cDelta"] == cDelta
        assert hms["log10MDelta"] == log10MDelta


def test_mass_definitions(mass_def):
    """Test the MassDefinition <-> Nc.HaloMassSummaryMassDef round-trip."""
    ncm_mass_def = mass_def.to_ncm()
    assert issubclass(type(ncm_mass_def), Nc.HaloMassSummaryMassDef)

    copy_mass_def = MassDefinition.from_ncm(ncm_mass_def)
    assert copy_mass_def == mass_def


def test_density_profiles(density_profile):
    """Test the DensityProfile <-> Nc.HaloDensityProfile round-trip."""
    hms = Nc.HaloCMParam.new(Nc.HaloMassSummaryMassDef.CRITICAL, 200.0)
    ncm_density_profile = density_profile.to_ncm()(hms)
    assert issubclass(type(ncm_density_profile), Nc.HaloDensityProfile)

    copy_density_profile = DensityProfile.from_ncm(ncm_density_profile)
    assert copy_density_profile == density_profile


def test_extract_mset_params_round_trip(mass_def, density_profile):
    """extract_mset_params -> rebuild_mset should preserve all scalar state."""
    H0 = 72.5
    OmegaB = 0.048
    OmegaC = 0.26
    w = -0.95
    Omegak = 0.01
    mass_delta = 250.0
    ra = 150.0
    dec = 2.0
    z = 0.4
    cDelta = 5.0
    log10MDelta = 14.2

    mset = create_mset(
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

    params = extract_mset_params(mset)
    assert params["H0"] == H0
    assert params["OmegaB"] == OmegaB
    assert params["OmegaC"] == OmegaC
    assert params["w"] == w
    assert params["Omegak"] == Omegak
    assert params["mass_def"] == mass_def
    assert params["mass_delta"] == mass_delta
    assert params["density_profile"] == density_profile
    assert params["ra"] == ra
    assert params["dec"] == dec
    assert params["z"] == z
    assert params["cDelta"] == cDelta
    assert params["log10MDelta"] == log10MDelta

    rebuilt = rebuild_mset(params)
    assert isinstance(rebuilt, Ncm.MSet)
    assert extract_mset_params(rebuilt) == params
