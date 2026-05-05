"""
Test suite for the utility functions in the `utils` module.
"""

from src.utils.utils import (
    create_ncm_spline,
    compute_radius,
    compute_tangential_component,
    compute_cross_component,
    CoordSystem,
)
from typing import Any, Tuple, cast
import numpy as np
import pytest
from numcosmo_py import Ncm, Nc
from numpy.typing import NDArray


@pytest.fixture(
    name="valid_pz_and_nodes",
    params=[
        "gaussian",
        "gaussian_narrow",
        "uniform",
        "linear",
    ],
)
def fixture_valid_pz_and_nodes(
    request,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fixture to provide different valid P(z) and nodes for testing the
    `create_ncm_spline` function. The P(z) and nodes will be generated according to the
    specified type.
    """
    if request.param == "gaussian":
        nodes = np.linspace(0, 3, 100)
        pz = np.exp(-0.5 * ((nodes - 1.5) / 0.5) ** 2)
    elif request.param == "gaussian_narrow":
        nodes = np.linspace(0, 3, 100)
        pz = np.exp(-0.5 * ((nodes - 1.5) / 0.005) ** 2)
    elif request.param == "uniform":
        nodes = np.linspace(0, 3, 100)
        pz = np.ones_like(nodes)
    elif request.param == "linear":
        nodes = np.linspace(0, 3, 100)
        pz = np.maximum(0, 1 - nodes / 2)
    else:  # pragma no cover
        raise ValueError("Invalid test case for valid_pz_and_nodes")

    return pz, nodes


@pytest.fixture(
    name="invalid_pz_and_nodes",
    params=[
        "different_lengths",
        "not_arrays",
    ],
)
def fixture_invalid_pz_and_nodes(
    request,
) -> Tuple[Any, Any]:
    """
    Fixture to provide different invalid P(z) and nodes for testing the
    `create_ncm_spline` function. The invalid cases include different lengths of P(z)
    and nodes, and non-array inputs.
    """
    if request.param == "different_lengths":
        pz = np.array([0.1, 0.2, 0.3])
        nodes = np.array([0.0, 1.0])
    elif request.param == "not_arrays":
        pz = [0.1, 0.2, 0.3]
        nodes = [0.0, 1.0, 2.0]
    else:  # pragma no cover
        raise ValueError("Invalid test case for invalid_pz_and_nodes")

    return pz, nodes


@pytest.fixture(name="valid_low_compute_radius")
def fixture_valid_low_compute_radius() -> Tuple[NDArray, NDArray, Ncm.MSet]:
    """
    Fixture to provide valid inputs for testing the `compute_radius` function. The
    fixture generates random right ascension, declination, and a cosmological model for
    each test case.
    """
    ra_cl = np.random.uniform(-180, 180)
    dec_cl = np.random.uniform(-90, 90)
    z_cl = np.random.uniform(0, 3)
    ra = ra_cl + np.random.uniform(-0.1, 0.1, size=100)
    dec = dec_cl + np.random.uniform(-0.1, 0.1, size=100)

    cosmo = Nc.HICosmoDEXcdm()
    prim = Nc.HIPrimPowerLaw.new()
    tf = Nc.TransferFuncEH()
    psml = Nc.PowspecMLTransfer.new(tf)
    reion = Nc.HIReionCamb.new()

    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    dist = Nc.Distance.new(6.0)
    hp = Nc.HaloPosition.new(dist)

    hp["ra"] = ra_cl
    hp["dec"] = dec_cl
    hp["z"] = z_cl

    hp.prepare(cosmo)

    mset = Ncm.MSet.new_array(
        [
            cosmo,
            hp,
        ]
    )

    return ra, dec, mset


@pytest.fixture(name="valid_high_compute_radius")
def fixture_valid_high_compute_radius() -> Tuple[NDArray, NDArray, Ncm.MSet]:
    """
    Fixture to provide valid inputs for testing the `compute_radius` function. The
    fixture generates random right ascension, declination, and a cosmological model for
    each test case, with higher redshift values to test the function's behavior at
    larger distances.
    """
    ra_cl = np.random.uniform(-180, 180)
    dec_cl = np.random.uniform(-90, 90)
    z_cl = np.random.uniform(0, 3)
    ra = np.empty(100)
    dec = np.empty(100)

    for i in range(100):
        while True:
            ra[i] = ra_cl + np.random.uniform(-5, 5)
            dec[i] = dec_cl + np.random.uniform(-5, 5)

            if abs(ra[i] - ra_cl) > 2 or abs(dec[i] - dec_cl) > 2:
                break

    cosmo = Nc.HICosmoDEXcdm()
    prim = Nc.HIPrimPowerLaw.new()
    tf = Nc.TransferFuncEH()
    psml = Nc.PowspecMLTransfer.new(tf)
    reion = Nc.HIReionCamb.new()

    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    dist = Nc.Distance.new(6.0)
    hp = Nc.HaloPosition.new(dist)

    hp["ra"] = ra_cl
    hp["dec"] = dec_cl
    hp["z"] = z_cl

    hp.prepare(cosmo)

    mset = Ncm.MSet.new_array(
        [
            cosmo,
            hp,
        ]
    )

    return ra, dec, mset


@pytest.fixture(
    name="invalid_compute_radius_ra_dec", params=["invalid_ra", "invalid_dec"]
)
def fixture_invalid_compute_radius_ra_dec(request) -> Tuple[Any, Any, Any]:
    """
    Fixture to provide invalid inputs for testing the `compute_radius` function. The
    fixture generates random right ascension, declination, and a cosmological model for
    each test case, with some cases having invalid types to test the function's error
    handling.
    """
    ra_cl = np.random.uniform(-180, 180)
    dec_cl = np.random.uniform(-90, 90)
    z_cl = np.random.uniform(0, 3)

    cosmo = Nc.HICosmoDEXcdm()
    prim = Nc.HIPrimPowerLaw.new()
    tf = Nc.TransferFuncEH()
    psml = Nc.PowspecMLTransfer.new(tf)
    reion = Nc.HIReionCamb.new()

    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    dist = Nc.Distance.new(6.0)
    hp = Nc.HaloPosition.new(dist)

    hp["ra"] = ra_cl
    hp["dec"] = dec_cl
    hp["z"] = z_cl

    hp.prepare(cosmo)

    mset = Ncm.MSet.new_array(
        [
            cosmo,
            hp,
        ]
    )

    if "invalid_ra" in request.param:
        ra = "invalid_ra"
        dec = dec_cl
    else:
        ra = ra_cl
        dec = "invalid_dec"

    return ra, dec, mset


@pytest.fixture(
    name="invalid_compute_radius_mset",
    params=["invalid_mset", "no_cosmo", "no_hp"],
)
def fixture_invalid_compute_radius_mset(request) -> Tuple[float, float, Any]:
    """
    Fixture to provide invalid inputs for testing the `compute_radius` function. The
    fixture generates random right ascension, declination, and an invalid cosmological
    model to test the function's error handling.
    """
    ra_cl = np.random.uniform(-180, 180)
    dec_cl = np.random.uniform(-90, 90)
    z_cl = np.random.uniform(0, 3)

    cosmo = Nc.HICosmoDEXcdm()
    prim = Nc.HIPrimPowerLaw.new()
    tf = Nc.TransferFuncEH()
    psml = Nc.PowspecMLTransfer.new(tf)
    reion = Nc.HIReionCamb.new()

    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    dist = Nc.Distance.new(6.0)
    hp = Nc.HaloPosition.new(dist)

    hp["ra"] = ra_cl
    hp["dec"] = dec_cl
    hp["z"] = z_cl

    hp.prepare(cosmo)

    if "invalid_mset" in request.param:
        mset = "invalid_mset"
    elif "no_cosmo" in request.param:
        mset = Ncm.MSet.new_array([hp])
    elif "no_hp" in request.param:
        mset = Ncm.MSet.new_array([cosmo])
    else:  # pragma no cover
        raise ValueError("Invalid test case for invalid_compute_radius_mset")

    return ra_cl, dec_cl, mset


@pytest.fixture(
    name="valid_ellipticity",
    params=[CoordSystem.CELESTIAL, CoordSystem.EUCLIDEAN],
)
def fixture_valid_ellipticity(
    request,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, CoordSystem, Ncm.MSet]:
    """
    Fixture to provide valid inputs for testing the `compute_tangential_component`
    function. The fixture generates random shape parameters, right ascension, declination,
    coordinate system, and a cosmological model for each test case.
    """
    ra_cl = np.random.uniform(-180, 180)
    dec_cl = np.random.uniform(-90, 90)
    ra = ra_cl + np.random.uniform(-0.1, 0.1, size=100)
    dec = dec_cl + np.random.uniform(-0.1, 0.1, size=100)
    e1 = np.random.uniform(-0.5, 0.5, size=100)
    e2 = np.random.uniform(-0.5, 0.5, size=100)
    z_cl = np.random.uniform(0, 3)

    cosmo = Nc.HICosmoDEXcdm()
    prim = Nc.HIPrimPowerLaw.new()
    tf = Nc.TransferFuncEH()
    psml = Nc.PowspecMLTransfer.new(tf)
    reion = Nc.HIReionCamb.new()

    cosmo.params_set_default_ftype()
    cosmo.omega_x2omega_k()
    psml.require_kmin(1.0e-6)
    psml.require_kmax(1.0e3)
    cosmo.add_submodel(prim)
    cosmo.add_submodel(reion)

    dist = Nc.Distance.new(6.0)
    hp = Nc.HaloPosition.new(dist)

    hp["ra"] = ra_cl
    hp["dec"] = dec_cl
    hp["z"] = z_cl

    hp.prepare(cosmo)

    mset = Ncm.MSet.new_array(
        [
            cosmo,
            hp,
        ]
    )

    return e1, e2, ra, dec, request.param, mset


def test_create_ncm_spline_valid(
    valid_pz_and_nodes: Tuple[NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """
    Test the `create_ncm_spline` function with valid P(z) and nodes. The test checks if
    the function returns a valid Ncm.Spline object without raising any exceptions.
    """
    pz, nodes = valid_pz_and_nodes
    spline = create_ncm_spline(pz, nodes)
    assert isinstance(spline, Ncm.Spline)
    assert spline.get_len() <= len(nodes)


def test_create_ncm_spline_invalid(
    invalid_pz_and_nodes: Tuple[Any, Any],
) -> None:
    """
    Test the `create_ncm_spline` function with invalid P(z) and nodes. The test checks
    if the function raises a ValueError when the lengths of P(z) and nodes are
    different or when the inputs are not numpy arrays.
    """
    pz, nodes = invalid_pz_and_nodes
    with pytest.raises(ValueError):
        create_ncm_spline(pz, nodes)


def test_compute_radius_low(
    valid_low_compute_radius: Tuple[NDArray, NDArray, Ncm.MSet],
) -> None:
    """
    Test the `compute_radius` function with valid inputs. The test checks if the
    function returns a non-negative radius without raising any exceptions.
    """
    ra, dec, mset = valid_low_compute_radius
    for ra_i, dec_i in zip(ra, dec):
        radius = compute_radius(ra_i, dec_i, mset)
        assert 0 <= radius < 5


def test_compute_radius_high(
    valid_high_compute_radius: Tuple[NDArray, NDArray, Ncm.MSet],
) -> None:
    """
    Test the `compute_radius` function with valid inputs. The test checks if the
    function returns a non-negative radius without raising any exceptions.
    """
    ra, dec, mset = valid_high_compute_radius
    for ra_i, dec_i in zip(ra, dec):
        radius = compute_radius(ra_i, dec_i, mset)
        assert radius >= 5


def test_compute_radius_invalid_ra_dec(
    invalid_compute_radius_ra_dec: Tuple[Any, Any, Any],
) -> None:
    """
    Test the `compute_radius` function with invalid right ascension and declination. The
    test checks if the function raises a ValueError when the inputs are not valid
    numbers.
    """
    ra, dec, mset = invalid_compute_radius_ra_dec
    with pytest.raises(TypeError):
        compute_radius(ra, dec, mset)


def test_compute_radius_invalid_mset(
    invalid_compute_radius_mset: Tuple[float, float, Any],
) -> None:
    """
    Test the `compute_radius` function with invalid cosmological model. The test checks
    if the function raises a ValueError when the MSet does not contain a valid cosmological
    model or halo position, or when the halo position is not prepared.
    """
    ra, dec, mset = invalid_compute_radius_mset

    with pytest.raises((AttributeError, TypeError)):
        compute_radius(ra, dec, mset)


def test_compute_tangential_component_valid(
    valid_ellipticity: Tuple[NDArray, NDArray, NDArray, NDArray, CoordSystem, Ncm.MSet],
) -> None:
    """
    Test the `compute_tangential_component` function with valid inputs. The test checks
    if the function returns a tangential component without raising any exceptions.
    """
    e1, e2, ra, dec, coord_system, mset = valid_ellipticity

    for e1_i, e2_i, ra_i, dec_i in zip(e1, e2, ra, dec):
        tangential_component = compute_tangential_component(
            e1_i, e2_i, ra_i, dec_i, coord_system, mset
        )
        assert isinstance(tangential_component, float)

    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    hp["ra"] = 0.0
    hp["dec"] = 0.0

    hp.prepare(cosmo)

    known_e1 = [0.0, 0.5]
    known_e2 = [0.5, 0.0]
    known_ra = 0.1
    known_dec = 0.0

    expected_et = [0.0, -0.5]

    for i, (e1_i, e2_i) in enumerate(zip(known_e1, known_e2)):
        et = compute_tangential_component(
            e1_i, e2_i, known_ra, known_dec, coord_system, mset
        )
        assert np.isclose(et, expected_et[i], atol=1e-5)


def test_compute_cross_component_valid(
    valid_ellipticity: Tuple[NDArray, NDArray, NDArray, NDArray, CoordSystem, Ncm.MSet],
) -> None:
    """
    Test the `compute_cross_component` function with valid inputs. The test checks if
    the function returns a cross component without raising any exceptions.
    """
    e1, e2, ra, dec, coord_system, mset = valid_ellipticity

    for e1_i, e2_i, ra_i, dec_i in zip(e1, e2, ra, dec):
        cross_component = compute_cross_component(
            e1_i, e2_i, ra_i, dec_i, coord_system, mset
        )
        assert isinstance(cross_component, float)

    cosmo = cast(Nc.HICosmo, mset.peek_by_name("NcHICosmo"))
    hp = cast(Nc.HaloPosition, mset.peek_by_name("NcHaloPosition"))
    hp["ra"] = 0.0
    hp["dec"] = 0.0

    hp.prepare(cosmo)

    known_e1 = [0.0, 0.5]
    known_e2 = [0.5, 0.0]
    known_ra = 0.1
    known_dec = 0.1

    expected_ex = [0.0, 0.5]

    for i, (e1_i, e2_i) in enumerate(zip(known_e1, known_e2)):
        ex = compute_cross_component(
            e1_i, e2_i, known_ra, known_dec, coord_system, mset
        )
        if coord_system == CoordSystem.CELESTIAL:
            assert np.isclose(ex, expected_ex[i], atol=1e-5)
        else:
            assert np.isclose(ex, -expected_ex[i], atol=1e-5)
