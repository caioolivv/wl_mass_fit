"""
Loader for numcosmo experiment files.

A numcosmo experiment is a pair of files in the same directory:

  ``<stem>.yaml``                # serialized model-set and likelihood
  ``<stem>.dataset.gvar``        # binary serialization of the dataset referenced by the YAML

The YAML refers to the dataset via an anchor (e.g. ``*S0``) that is only defined
inside the ``.dataset.gvar`` file, so the binary file must be deserialized first
with the same ``Ncm.Serialize`` instance.

The loader returns a :class:`LoadedExperiment` carrying the native numcosmo
objects (``mset``, ``likelihood``, ``dataset``) plus a pandas DataFrame in the
column layout that the binned/wtg likelihoods in :mod:`wl_mass_fit.likelihood`
already consume.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numcosmo_py import Nc, Ncm

from .utils.utils import CoordSystem


class ShapeModel(StrEnum):
    """Shape model used by a numcosmo experiment."""

    HSM_GAUSS = "hsm_gauss"
    HSM_GAUSS_GLOBAL = "hsm_gauss_global"


_SHAPE_MODEL_BY_TYPE: dict[type, ShapeModel] = {
    Nc.GalaxySDShapeHSMGauss: ShapeModel.HSM_GAUSS,
    Nc.GalaxySDShapeHSMGaussGlobal: ShapeModel.HSM_GAUSS_GLOBAL,
}


@dataclass(frozen=True)
class LoadedExperiment:
    """Container for a fully loaded numcosmo experiment."""

    mset: Ncm.MSet
    likelihood: Ncm.Likelihood
    dataset: Ncm.Dataset
    obs: pd.DataFrame
    coord_system: CoordSystem
    shape_model: ShapeModel
    yaml_path: Path
    dataset_path: Path


def _detect_shape_model(mset: Ncm.MSet) -> ShapeModel:
    sd = mset.peek_by_name("NcGalaxySDShape")
    for ncm_cls, sm in _SHAPE_MODEL_BY_TYPE.items():
        if isinstance(sd, ncm_cls):
            return sm
    raise ValueError(
        f"Unsupported NcGalaxySDShape subclass: {type(sd).__name__}. "
        f"Supported: {[c.__name__ for c in _SHAPE_MODEL_BY_TYPE]}"
    )


def _wl_obs_to_dataframe(
    wl_obs: Nc.GalaxyWLObs, shape_model: ShapeModel
) -> pd.DataFrame:
    """Materialize an Nc.GalaxyWLObs into a DataFrame matching REQUIRED_COLUMNS."""
    available = list(wl_obs.peek_columns())
    n = wl_obs.len()

    column_map = {
        "ra": "i_ra",
        "dec": "i_dec",
        "epsilon_obs_1": "i_hsmshaperegauss_e1",
        "epsilon_obs_2": "i_hsmshaperegauss_e2",
        "std_shape": "i_hsmshaperegauss_derived_rms_e",
        "std_noise": "i_hsmshaperegauss_derived_sigma_e",
        "m": "i_hsmshaperegauss_derived_shear_bias_m",
        "c1": "i_hsmshaperegauss_derived_shear_bias_c1",
        "c2": "i_hsmshaperegauss_derived_shear_bias_c2",
        "z": "z",
    }

    df = pd.DataFrame()
    for src, dst in column_map.items():
        if src in available:
            df[dst] = [wl_obs.get(src, i) for i in range(n)]
        elif src == "std_shape" and shape_model is ShapeModel.HSM_GAUSS_GLOBAL:
            df[dst] = np.zeros(n, dtype=np.float64)
        else:
            raise ValueError(
                f"Required wl_obs column '{src}' missing for shape model "
                f"{shape_model}. Available: {available}"
            )

    pz_weights: list[np.ndarray] = []
    pz_nodes: list[np.ndarray] = []
    for i in range(n):
        sp = wl_obs.peek_pz(i)
        pz_nodes.append(np.asarray(sp.peek_xv().dup_array(), dtype=np.float64))
        pz_weights.append(np.asarray(sp.peek_yv().dup_array(), dtype=np.float64))
    df["pz_weights"] = pz_weights
    df["pz_nodes"] = pz_nodes

    return df


def load_experiment(yaml_path: str | Path) -> LoadedExperiment:
    """
    Load a numcosmo experiment from disk.

    Parameters
    ----------
    yaml_path
        Path to the experiment YAML. The matching ``.dataset.gvar`` file is
        expected next to it (same stem, ``.dataset.gvar`` suffix).
    """
    yaml_path = Path(yaml_path).resolve()
    dataset_path = yaml_path.with_suffix(".dataset.gvar")
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    Ncm.cfg_init()

    ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
    dataset = cast(Ncm.Dataset, ser.from_binfile(str(dataset_path)))
    exp = ser.dict_str_from_yaml_file(str(yaml_path))
    mset = cast(Ncm.MSet, exp.get("model-set"))
    likelihood = cast(Ncm.Likelihood, exp.get("likelihood"))

    shape_model = _detect_shape_model(mset)
    wl_obs = dataset.get_data(0).peek_obs()
    coord_system = CoordSystem.from_ncm(Nc.GalaxyWLObsCoord(wl_obs.get_coord()))
    obs = _wl_obs_to_dataframe(wl_obs, shape_model)

    return LoadedExperiment(
        mset=mset,
        likelihood=likelihood,
        dataset=dataset,
        obs=obs,
        coord_system=coord_system,
        shape_model=shape_model,
        yaml_path=yaml_path,
        dataset_path=dataset_path,
    )
