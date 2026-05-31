"""Monte Carlo MLE analysis."""

from typing import Dict
from pydantic import ConfigDict, Field
import numpy as np
import pandas as pd

from ..likelihood.likelihood import Likelihood
from .analysis import MCAnalysis


class MCMLE(MCAnalysis):
    """
    Monte Carlo analysis of maximum likelihood estimates.

    Fits each mock with Likelihood.maximum_likelihood_estimate() and records
    the point estimates. Tracks bias and spread of point estimates.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    fparams_bounds: Dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Bounds for fit parameters {fparam_name: (lower, upper)}.",
    )
    boundary_margin: float = Field(
        default=1e-2,
        ge=0.0,
        lt=0.5,
        description=(
            "Fraction of each parameter's bound range that counts as 'pinned to "
            "the boundary'. A fit landing within margin*(upper-lower) of either "
            "bound is flagged as a failure: when the optimizer rails against a "
            "bound it settles just inside it, not exactly on it. Only used when "
            "fparams_bounds is provided. Set to 0 to accept any in-bounds value."
        ),
    )

    def _fit_one(self, lik: Likelihood) -> Dict:
        """Fit with maximum likelihood estimate."""
        theta = lik.maximum_likelihood_estimate()
        row = {name: val for name, val in zip(lik.fparams, theta)}
        return row

    def _is_successful(self, row: Dict) -> bool:
        """A fit succeeds when every recorded parameter is finite and, for
        parameters with bounds, sits clear of both boundaries.

        Boundary-pinned fits are rejected via an absolute margin scaled to each
        parameter's own bound range (``boundary_margin * (upper - lower)``),
        rather than the scale-coupled ``lower * 1.05 < x < upper * 0.95`` test
        it replaces.
        """
        if not all(
            np.isfinite(v) for v in row.values() if isinstance(v, (int, float))
        ):
            return False

        if self.fparams_bounds is None:
            return True

        for fparam, (lower, upper) in self.fparams_bounds.items():
            if fparam not in row:
                continue
            margin = self.boundary_margin * (upper - lower)
            if not (lower + margin < row[fparam] < upper - margin):
                return False
        return True

    def _config_metadata(self) -> Dict:
        """Return config for resumability checks."""
        meta = {
            "C_NGALS": self.generator.n_gals,
            "C_RMIN": self.generator.r_min,
            "C_RMAX": self.generator.r_max,
            "C_RMISC": self.generator.r_miscenter,
            "C_TRA": self.generator.true_ra,
            "C_TDEC": self.generator.true_dec,
        }
        return meta

    def _aggregate_results(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Aggregate MLE results: bias, spread, quantiles."""
        successful_df = df[df.get("success", True) == True]

        per_param = {}
        for col in df.columns:
            if col in ["fit_time", "success", "index"]:
                continue

            values = successful_df[col].to_numpy()
            if len(values) == 0:
                continue

            truth = self._truth_for(col)

            stats = {
                "mean": np.mean(values),
                "std": np.std(values, ddof=1) if len(values) > 1 else 0.0,
                "median": np.median(values),
                "q05": np.quantile(values, 0.05),
                "q16": np.quantile(values, 0.16),
                "q50": np.quantile(values, 0.50),
                "q84": np.quantile(values, 0.84),
                "q95": np.quantile(values, 0.95),
            }

            if np.isfinite(truth):
                stats["bias"] = stats["mean"] - truth

            per_param[col] = stats

        return per_param
