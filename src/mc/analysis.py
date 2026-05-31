"""Base Monte Carlo analysis class and result container."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict
import json
import time
import warnings

import numpy as np
import pandas as pd
from numcosmo_py import Ncm
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..likelihood.likelihood import Likelihood
from ..mock.generator import MockGenerator


@dataclass(frozen=True)
class MCResult:
    """Container for Monte Carlo analysis summary statistics."""

    n_total: int
    n_successful: int
    success_rate: float
    per_param: Dict[str, Dict[str, float]]


class MCAnalysis(BaseModel, ABC):
    """
    Base class for Monte Carlo analyses of mass fitting.

    Drives the iteration loop (generate mock → fit → record) with resumability.
    Subclasses implement only the per-iteration fit step and success criterion.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    generator: MockGenerator = Field(
        description="Configured mock generator."
    )
    likelihood_factory: Callable[[pd.DataFrame], Likelihood] = Field(
        description="Callable that takes obs DataFrame and returns a Likelihood instance."
    )
    n_iter: int = Field(
        description="Target number of successful iterations."
    )
    output: Path = Field(
        description="Output FITS file path."
    )
    true_params: Dict[str, float] = Field(
        default_factory=dict,
        description="Truth parameters {fparam_full_name: value} for diagnostics.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for NumCosmo RNG.",
    )
    verbose: bool = Field(
        default=False,
        description="Print per-iteration progress to stdout.",
    )

    def _truth_for(self, key: str) -> float:
        """Resolve truth value for *key* from true_params.

        Tries exact match first, then short name (after ``:``) match, so both
        ``"NcHaloMassSummary:log10MDelta"`` and ``"log10MDelta"`` resolve
        against the same dict regardless of which convention the caller uses.
        """
        if key in self.true_params:
            return self.true_params[key]
        short = key.split(":", 1)[-1]
        for k, v in self.true_params.items():
            if k == short or k.split(":", 1)[-1] == short:
                return v
        return float("nan")

    @field_validator("n_iter")
    @classmethod
    def validate_n_iter(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_iter must be positive.")
        return v

    @field_validator("output", mode="before")
    @classmethod
    def validate_output(cls, v) -> Path:
        return Path(v)

    @abstractmethod
    def _fit_one(self, lik: Likelihood) -> Dict:
        """
        Fit one iteration.

        Parameters
        ----------
        lik : Likelihood
            Prepared likelihood instance.

        Returns
        -------
        dict
            Row data (fparam values, possibly more). Must not include
            'fit_time' or 'success' (added by base class).
        """
        raise NotImplementedError

    @abstractmethod
    def _is_successful(self, row: Dict) -> bool:
        """
        Determine if a fit row is successful.

        Parameters
        ----------
        row : dict
            Output from _fit_one.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @abstractmethod
    def _config_metadata(self) -> Dict:
        """
        Return metadata dict for resumability consistency checks.

        Returns
        -------
        dict
            Key-value pairs stored alongside output parquet.
        """
        raise NotImplementedError

    @property
    def _meta_path(self) -> Path:
        return self.output.with_suffix(self.output.suffix + ".meta.json")

    def _write_results(self, df: pd.DataFrame, config: Dict) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output, index=False)
        self._meta_path.write_text(json.dumps(config))

    def _load_existing(self, config: Dict):
        """Load existing parquet + sidecar; return (df, n_successful)."""
        if not self.output.exists():
            return pd.DataFrame(), 0
        try:
            existing_df = pd.read_parquet(self.output)
            if self._meta_path.exists():
                stored = json.loads(self._meta_path.read_text())
                mismatches = [
                    (k, stored[k], v)
                    for k, v in config.items()
                    if k in stored and stored[k] != v
                ]
                if mismatches:
                    raise ValueError(f"Configuration mismatch: {mismatches}")
            n_successful = int((existing_df["success"] == True).sum()) if "success" in existing_df.columns else 0
            return existing_df, n_successful
        except ValueError:
            raise
        except Exception as e:
            warnings.warn(f"Could not load existing results: {e}")
            return pd.DataFrame(), 0

    def run(self) -> pd.DataFrame:
        """
        Run the Monte Carlo loop.

        Generates mocks, fits each, records results to FITS, resumes from
        existing file if present.

        Returns
        -------
        pd.DataFrame
            All iterations (existing + new).
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Initialize NumCosmo
        Ncm.cfg_init()
        Ncm.cfg_set_log_handler(lambda msg: None)

        # Initialize RNG
        if self.seed is not None:
            np.random.seed(self.seed)
            rng = Ncm.RNG.seeded_new(None, self.seed)
        else:
            rng = Ncm.RNG.seeded_new(None, np.random.randint(0, 2**32 - 1))

        # Check for existing results
        config = self._config_metadata()
        existing_df, n_successful = self._load_existing(config)

        # Save the initial generator mset param values so each iteration starts
        # from the true model (NumCosmo retains param values after a fit).
        _gen_reset: list[tuple[str, str, float]] = []
        for full_name in self.true_params:
            if ":" in full_name:
                model_name, param_name = full_name.split(":", 1)
                model = self.generator.mset.peek_by_name(model_name)
                if model is not None:
                    _gen_reset.append((model_name, param_name, model[param_name]))

        # Main loop
        all_results_df = existing_df.copy()
        n_remaining = self.n_iter - n_successful

        if n_remaining <= 0:
            if self.verbose:
                print(
                    f"✓ Already have {n_successful} successful results, nothing to do."
                )
            return all_results_df

        if self.verbose:
            print(f"Running {n_remaining} iterations ({n_successful} done)...\n")

        idx = 0
        while idx < n_remaining:
            # Restore generator mset to true param values before each generate.
            for model_name, param_name, value in _gen_reset:
                self.generator.mset.peek_by_name(model_name)[param_name] = value

            # Generate mock
            df = self.generator.generate(rng)

            # Create and prepare likelihood
            lik = self.likelihood_factory(df)
            lik.prepare_data()

            # Fit
            start_time = time.time()
            row = self._fit_one(lik)
            fit_time = time.time() - start_time

            # Add metadata
            row["fit_time"] = fit_time
            row["success"] = self._is_successful(row)

            # Record
            if row["success"]:
                idx += 1

            all_results_df = pd.concat(
                [all_results_df, pd.DataFrame([row])], ignore_index=True
            )

            # Save parquet + sidecar
            self._write_results(all_results_df, config)

            # Print progress
            success_rate = (
                ((idx + n_successful) / len(all_results_df) * 100)
                if len(all_results_df) > 0
                else 0.0
            )
            if self.verbose:
                status = "✓" if row["success"] else "✗"
                print(
                    f"Iteration {len(all_results_df)}: {status} "
                    f"({idx + n_successful}/{self.n_iter}, "
                    f"{success_rate:.1f}%, {fit_time:.3f}s)"
                )

        return all_results_df

    def summary(self) -> MCResult:
        """
        Compute summary statistics from results file.

        Returns
        -------
        MCResult
            Aggregated statistics.
        """
        if not self.output.exists():
            raise FileNotFoundError(f"Output file not found: {self.output}")

        df = pd.read_parquet(self.output)
        if "success" not in df.columns:
            df["success"] = True

        n_total = len(df)
        n_successful = (df["success"] == True).sum()
        success_rate = n_successful / n_total if n_total > 0 else 0.0

        # Subclass-specific aggregation
        per_param = self._aggregate_results(df)

        return MCResult(
            n_total=n_total,
            n_successful=n_successful,
            success_rate=success_rate,
            per_param=per_param,
        )

    @abstractmethod
    def _aggregate_results(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Aggregate results into per-parameter statistics.

        Parameters
        ----------
        df : pd.DataFrame
            Full results DataFrame.

        Returns
        -------
        dict
            {fparam_name: {metric: value, ...}, ...}
        """
        raise NotImplementedError
