"""
Galaxy property sources for mock generation, with adapters for real data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import cast, Optional
import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from numcosmo_py import Ncm, Nc

from ..utils.utils import create_ncm_spline


@dataclass(frozen=True)
class GalaxyProperties:
    """Per-galaxy record: redshift, P(z), shear-bias, and shape parameters."""

    z: float

    m: float
    c1: float
    c2: float
    sigma_e: float  # std_noise: per-galaxy measurement error
    rms_e: float  # std_shape: intrinsic shape dispersion
    pz_spline: Optional[Ncm.Spline] = (
        None  # pre-built Ncm.Spline; None for non-Hamana sources
    )
    pz_nodes: Optional[NDArray[np.float64]] = None
    pz_weights: Optional[NDArray[np.float64]] = None


@dataclass(frozen=True)
class _ClusterEntry:
    """Internal: a cluster redshift and its galaxy properties."""

    z_cluster: float
    galaxies: list[GalaxyProperties]


class GalaxySource(ABC):
    """Abstract interface for sampling galaxy properties."""

    @abstractmethod
    def sample(self, z_target: float, rng: np.random.Generator) -> GalaxyProperties:
        """
        Sample a single galaxy's properties at a target cluster redshift.

        Parameters
        ----------
        z_target : float
            Target redshift to find similar clusters.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        GalaxyProperties
            Sampled galaxy properties.
        """
        raise NotImplementedError


class _BaseGalaxySource(GalaxySource):
    """Base implementation sharing the redshift-weighted sampling logic."""

    def __init__(self, clusters: list[_ClusterEntry], z_tol: float = 0.1):
        self._clusters = clusters
        self._z_tol = z_tol

    def redshift_counts(self) -> list[tuple[float, int]]:
        """Return (z_cluster, n_galaxies) for every loaded cluster."""
        return [(c.z_cluster, len(c.galaxies)) for c in self._clusters]

    def sample(self, z_target: float, rng: np.random.Generator) -> GalaxyProperties:
        """Sample weighted by inverse redshift distance, with fallback."""
        # Find clusters within tolerance
        nearby = [
            c for c in self._clusters if abs(c.z_cluster - z_target) <= self._z_tol
        ]

        # Fallback to 5 nearest if empty
        if not nearby:
            sorted_clusters = sorted(
                self._clusters, key=lambda c: abs(c.z_cluster - z_target)
            )
            nearby = sorted_clusters[:5]

        # Weight by inverse Δz
        weights = np.array([1.0 / (abs(c.z_cluster - z_target) + 0.01) for c in nearby])
        weights /= weights.sum()

        cluster_idx = rng.choice(len(nearby), p=weights)
        cluster = nearby[cluster_idx]

        gal_idx = rng.integers(0, len(cluster.galaxies))
        return cluster.galaxies[gal_idx]


class DataFrameGalaxySource(GalaxySource):
    """
    Source that draws galaxies from one DataFrame.
    """

    def __init__(self, obs_df: pd.DataFrame, z_cluster: float):
        """
        Parameters
        ----------
        obs_df : pd.DataFrame
            DataFrame containing galaxy properties and P(z) info for a single cluster.
        z_cluster : float
            Redshift of the cluster corresponding to the DataFrame.
        """
        galaxies = []
        for _, row in obs_df.iterrows():
            pz_spline = (
                cast(Ncm.Spline, row["pz_spline"]) if "pz_spline" in row else None
            )
            gal = GalaxyProperties(
                z=float(row["z"]),
                m=float(row["i_hsmshaperegauss_derived_shear_bias_m"]),
                c1=float(row["i_hsmshaperegauss_derived_shear_bias_c1"]),
                c2=float(row["i_hsmshaperegauss_derived_shear_bias_c2"]),
                sigma_e=float(row["i_hsmshaperegauss_derived_sigma_e"]),
                rms_e=float(row["i_hsmshaperegauss_derived_rms_e"]),
                pz_spline=pz_spline,
                pz_nodes=np.asarray(row["pz_nodes"]),
                pz_weights=np.asarray(row["pz_weights"]),
            )
            galaxies.append(gal)
        cluster = _ClusterEntry(z_cluster=z_cluster, galaxies=galaxies)
        self._clusters = cluster

    def sample(self, z_target: float, rng: np.random.Generator) -> GalaxyProperties:
        """
        Sample a galaxy and also generate its position and shape in the Ncm.MSet.

        Parameters
        ----------
        z_target : float
            Target redshift to find similar clusters.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        GalaxyProperties
            Sampled galaxy properties.
        """
        gal_idx = rng.integers(0, len(self._clusters.galaxies))
        return self._clusters.galaxies[gal_idx]


class HamanaGalaxySource(_BaseGalaxySource):
    """
    Source that loads galaxies from legacy Hamana YAML + `.gvar` files.

    Expects directory structure: `HWL16a-*/`{`_experiment_pdf.yaml`,
    `_experiment_pdf.dataset.gvar`}
    """

    @classmethod
    def from_directory(
        cls,
        path: Path,
        suffix: str = "_experiment_pdf",
        z_tol: float = 0.1,
    ) -> "HamanaGalaxySource":
        """
        Load all Hamana cluster data from a directory.

        Parameters
        ----------
        path : Path
            Directory containing `HWL16a-*` subdirectories.
        suffix : str
            File suffix used when building dataset/YAML filenames, e.g.
            ``"_experiment_pdf"`` (default) or ``"_mass"``.
        z_tol : float
            Redshift tolerance for cluster selection.

        Returns
        -------
        HamanaGalaxySource
            The loaded source.
        """
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Invalid clusters path: {path}")

        cluster_dirs = sorted(
            [
                d
                for d in os.listdir(path)
                if d.startswith("HWL16a-") and (path / d).is_dir()
            ]
        )

        clusters = []

        for cluster_dir in cluster_dirs:
            yaml_file = path / cluster_dir / f"{cluster_dir}{suffix}.yaml"
            dataset_file = path / cluster_dir / f"{cluster_dir}{suffix}.dataset.gvar"

            if not (yaml_file.exists() and dataset_file.exists()):
                continue

            # Load observation data
            ser = Ncm.Serialize.new(Ncm.SerializeOpt.CLEAN_DUP)
            dataset = cast(
                Ncm.Dataset, ser.from_binfile(dataset_file.absolute().as_posix())
            )
            cluster_data = cast(Nc.DataClusterWL, dataset.get_data(0))
            wl_obs = cluster_data.peek_obs()

            # Load halo position
            experiment = ser.dict_str_from_yaml_file(yaml_file.absolute().as_posix())
            model_set = cast(Ncm.MSet, experiment.get("model-set"))
            halo_position = cast(
                Nc.HaloPosition, model_set.peek_by_name("NcHaloPosition")
            )

            z_cluster = float(halo_position["z"])

            # Extract galaxies from WLObs
            galaxies = []
            for j in range(wl_obs.len()):
                # Get per-galaxy properties
                gal_z = float(wl_obs.get("z", j))
                gal_m = float(wl_obs.get("m", j))
                gal_c1 = float(wl_obs.get("c1", j))
                gal_c2 = float(wl_obs.get("c2", j))
                std_shape = float(wl_obs.get("std_shape", j))
                std_noise = float(wl_obs.get("std_noise", j))

                # Extract P(z) spline and convert to nodes/weights
                pz_spline = wl_obs.peek_pz(j)
                pz_spline.prepare()

                # Extract knot vector and values
                xv = pz_spline.peek_xv()
                yv = pz_spline.peek_yv()

                pz_nodes = np.array([xv.get(i) for i in range(xv.len())])
                pz_weights = np.array([yv.get(i) for i in range(yv.len())])

                gal = GalaxyProperties(
                    z=gal_z,
                    pz_nodes=pz_nodes,
                    pz_weights=pz_weights,
                    m=gal_m,
                    c1=gal_c1,
                    c2=gal_c2,
                    sigma_e=std_noise,
                    rms_e=std_shape,
                    pz_spline=create_ncm_spline(pz_weights, pz_nodes),
                )
                galaxies.append(gal)

            clusters.append(_ClusterEntry(z_cluster=z_cluster, galaxies=galaxies))

        return cls(clusters, z_tol=z_tol)

    def __init__(self, clusters: list[_ClusterEntry], z_tol: float = 0.1):
        super().__init__(clusters, z_tol=z_tol)
