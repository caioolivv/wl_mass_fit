"""Monte Carlo posterior analysis."""

from typing import Dict
from pydantic import ConfigDict, Field
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from ..likelihood.likelihood import Likelihood
from .analysis import MCAnalysis


class MCPosterior(MCAnalysis):
    """
    Monte Carlo analysis of posterior distributions.

    Fits each mock with MCMC sampling and records posterior summary statistics.
    Tracks bias and spread of posterior point estimates, plus typical uncertainty size.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    nsamples: int = Field(
        description="Number of MCMC samples per iteration."
    )
    nwalkers: int = Field(
        default=32,
        description="Number of MCMC walkers."
    )
    nthreads: int = Field(
        default=1,
        description="Number of threads for MCMC."
    )
    burn_in: int = Field(
        default=0,
        description="Number of burn-in samples to discard."
    )
    progress: bool = Field(
        default=False,
        description="Display progress bar during MCMC."
    )
    fparams_bounds: Dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Bounds for MLE pre-screen {fparam_full_name: (lower, upper)}.",
    )

    def _mle_ok(self, mle_row: Dict) -> bool:
        """Return True if MLE result is finite and within fparams_bounds."""
        if self.fparams_bounds is None:
            return all(np.isfinite(v) for v in mle_row.values() if isinstance(v, (int, float)))
        for fparam, (lower, upper) in self.fparams_bounds.items():
            if fparam not in mle_row:
                continue
            if not np.isfinite(mle_row[fparam]) or not (lower < mle_row[fparam] < upper):
                return False
        return True

    def _nan_row(self, lik: Likelihood) -> Dict:
        """Return a row of NaN summary stats (signals failure to _is_successful)."""
        row = {}
        for full_name in lik.fparams:
            col = full_name.split(":", 1)[-1]
            for suf in ("_mean", "_median", "_std", "_q025", "_q16", "_q84", "_q975"):
                row[f"{col}{suf}"] = np.nan
        return row

    def _fit_one(self, lik: Likelihood) -> Dict:
        """Run MLE pre-screen, then MCMC if MLE succeeds."""
        theta = lik.maximum_likelihood_estimate()
        mle_row = {name: float(val) for name, val in zip(lik.fparams, theta)}
        if not self._mle_ok(mle_row):
            return self._nan_row(lik)

        with tempfile.TemporaryDirectory() as tmpdir:
            chain_file = Path(tmpdir) / "posterior.h5"
            chain = lik.sample_posterior(
                nsamples=self.nsamples,
                nwalkers=self.nwalkers,
                nthreads=self.nthreads,
                progress=self.progress,
                filename=str(chain_file),
            )

        chain = chain.iloc[self.burn_in:]

        row = {}
        for col in chain.columns:
            v = chain[col].to_numpy()
            row[f"{col}_mean"] = np.mean(v)
            row[f"{col}_median"] = np.median(v)
            row[f"{col}_std"] = np.std(v, ddof=1) if len(v) > 1 else 0.0
            row[f"{col}_q025"] = np.quantile(v, 0.025)
            row[f"{col}_q16"] = np.quantile(v, 0.16)
            row[f"{col}_q84"] = np.quantile(v, 0.84)
            row[f"{col}_q975"] = np.quantile(v, 0.975)

        return row

    def _is_successful(self, row: Dict) -> bool:
        """Check if posterior point estimates are finite."""
        for key in row:
            if "_mean" in key or "_median" in key:
                if not np.isfinite(row[key]):
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
            "C_NSAMP": self.nsamples,
            "C_NWALK": self.nwalkers,
        }
        return meta

    def _aggregate_results(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Aggregate posterior results: bias, spread, uncertainty size, coverage."""
        successful_df = df[df.get("success", True) == True]

        # Infer parameter names from columns
        param_names = set()
        for col in df.columns:
            if "_mean" in col:
                param_names.add(col.replace("_mean", ""))

        per_param = {}
        for param in param_names:
            mean_col = f"{param}_mean"
            median_col = f"{param}_median"
            std_col = f"{param}_std"
            q16_col = f"{param}_q16"
            q84_col = f"{param}_q84"
            q025_col = f"{param}_q025"
            q975_col = f"{param}_q975"

            if mean_col not in successful_df.columns:
                continue

            # Extract arrays
            means = successful_df[mean_col].to_numpy()
            medians = successful_df[median_col].to_numpy()
            stds = successful_df[std_col].to_numpy()
            q16s = successful_df[q16_col].to_numpy()
            q84s = successful_df[q84_col].to_numpy()
            q025s = successful_df[q025_col].to_numpy()
            q975s = successful_df[q975_col].to_numpy()

            truth = self._truth_for(param)

            # Bias of point estimates
            bias_mean = np.mean(means) - truth if np.isfinite(truth) else np.nan
            bias_median = np.median(medians) - truth if np.isfinite(truth) else np.nan

            # Spread of point estimates
            spread_mean = np.std(means, ddof=1) if len(means) > 1 else 0.0
            spread_median = np.std(medians, ddof=1) if len(medians) > 1 else 0.0

            # Typical uncertainty size
            uncertainty_std = np.mean(stds)
            uncertainty_1sigma = np.mean((q84s - q16s) / 2.0)
            uncertainty_2sigma = np.mean((q975s - q025s) / 2.0)

            # Coverage (fraction of iterations where interval contains truth)
            if np.isfinite(truth):
                coverage_1sigma = np.mean(
                    (q16s <= truth) & (truth <= q84s)
                )
                coverage_2sigma = np.mean(
                    (q025s <= truth) & (truth <= q975s)
                )
            else:
                coverage_1sigma = np.nan
                coverage_2sigma = np.nan

            per_param[param] = {
                "bias_mean": bias_mean,
                "bias_median": bias_median,
                "spread_mean": spread_mean,
                "spread_median": spread_median,
                "uncertainty_std": uncertainty_std,
                "uncertainty_1sigma": uncertainty_1sigma,
                "uncertainty_2sigma": uncertainty_2sigma,
                "coverage_1sigma": coverage_1sigma,
                "coverage_2sigma": coverage_2sigma,
            }

        return per_param
