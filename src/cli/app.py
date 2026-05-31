"""
wl-mass-fit CLI — four subcommands for MC and direct-fit analyses.
"""

from pathlib import Path
from typing import Annotated, Optional
import warnings

import typer

from ..likelihood.mset import MassDefinition, DensityProfile
from ..utils.utils import CoordSystem
from .builders import (
    LikelihoodType,
    build_mock_mset,
    build_mset,
    build_generator,
    build_likelihood_factory,
    parse_fparams,
    parse_fparams_bounds,
    parse_bin_edges,
    parse_radius_bounds,
)

app = typer.Typer(
    name="wl-mass-fit",
    help="Weak-lensing cluster mass fitting: Monte Carlo MLE/posterior and direct fitting.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Shared option annotations (reused across subcommands)
# ---------------------------------------------------------------------------

_RA = Annotated[float, typer.Option("--true-ra", help="True cluster RA (degrees).")]
_DEC = Annotated[float, typer.Option("--true-dec", help="True cluster Dec (degrees).")]
_LOG10M = Annotated[float, typer.Option("--log10m", help="True log10(M_delta) (solar masses).")]
_REDSHIFT = Annotated[float, typer.Option("--redshift", help="Cluster redshift.")]
_RMIN = Annotated[float, typer.Option("--r-min", help="Min sampling radius (Mpc/h).")]
_RMAX = Annotated[float, typer.Option("--r-max", help="Max sampling radius (Mpc/h).")]
_RMIS = Annotated[float, typer.Option("--r-miscenter", help="Miscentering offset (Mpc/h).")]
_NGALS = Annotated[int, typer.Option("--n-gals", help="Galaxies per mock resample.")]
_CLUSTERS = Annotated[Path, typer.Option("--clusters-path", help="Path to Hamana cluster data directory.")]
_OUTPUT = Annotated[Path, typer.Option("--output", help="Output parquet file path.")]
_NITER = Annotated[int, typer.Option("--n-iter", help="Target number of successful MC iterations.")]
_SEED = Annotated[Optional[int], typer.Option("--seed", help="Random seed.")]
_H0 = Annotated[float, typer.Option("--H0", help="Hubble constant (km/s/Mpc).")]
_OmB = Annotated[float, typer.Option("--OmegaB", help="Baryon density.")]
_OmC = Annotated[float, typer.Option("--OmegaC", help="CDM density.")]
_W = Annotated[float, typer.Option("--w", help="Dark energy EoS.")]
_OmK = Annotated[float, typer.Option("--Omegak", help="Curvature density.")]
_CDELTA = Annotated[float, typer.Option("--cDelta", help="NFW concentration.")]
_MASSDEF = Annotated[MassDefinition, typer.Option("--mass-def", help="Mass definition.")]
_MASSDELTA = Annotated[float, typer.Option("--mass-delta", help="Overdensity.")]
_PROFILE = Annotated[DensityProfile, typer.Option("--density-profile", help="Density profile.")]
_COORD = Annotated[CoordSystem, typer.Option("--coord-system", help="Coordinate system.")]
_PCUT = Annotated[float, typer.Option("--p-cut", help="Photo-z cumulative probability cut.")]
_DZ = Annotated[float, typer.Option("--delta-z", help="Photo-z catastrophic z threshold.")]
_ZMAX = Annotated[float, typer.Option("--z-max", help="Max individual galaxy redshift.")]
_LIK = Annotated[LikelihoodType, typer.Option("--likelihood", help="Likelihood subclass to use.")]
_FPARAMS = Annotated[Optional[str], typer.Option("--fparams", help="Comma-separated 'Model:param' fit params.")]
_FBOUNDS = Annotated[Optional[str], typer.Option("--fparams-bounds", help="Comma-sep 'lo:hi' bounds per fparam.")]
_BINEDGES = Annotated[Optional[str], typer.Option("--bin-edges", help="Comma-sep bin edges in Mpc (binned likelihoods).")]
_RADBOUNDS = Annotated[Optional[str], typer.Option("--radius-bounds", help="'lo,hi' Mpc bounds (wtg likelihood).")]


# ---------------------------------------------------------------------------
# mc-mle subcommand
# ---------------------------------------------------------------------------

@app.command("mc-mle")
def mc_mle(
    output: _OUTPUT,
    n_iter: _NITER,
    true_ra: _RA,
    true_dec: _DEC,
    log10m: _LOG10M,
    redshift: _REDSHIFT,
    r_min: _RMIN,
    r_max: _RMAX,
    r_miscenter: _RMIS,
    n_gals: _NGALS,
    clusters_path: _CLUSTERS,
    seed: _SEED = None,
    H0: _H0 = 70.0,
    OmegaB: _OmB = 0.045,
    OmegaC: _OmC = 0.255,
    w: _W = -1.0,
    Omegak: _OmK = 0.0,
    cDelta: _CDELTA = 4.0,
    mass_def: _MASSDEF = MassDefinition.CRITICAL,
    mass_delta: _MASSDELTA = 200.0,
    density_profile: _PROFILE = DensityProfile.NFW,
    coord_system: _COORD = CoordSystem.CELESTIAL,
    p_cut: _PCUT = 0.98,
    delta_z: _DZ = 0.2,
    z_max: _ZMAX = 2.5,
    likelihood: _LIK = LikelihoodType.BINNED_SHEAR,
    fparams: _FPARAMS = None,
    fparams_bounds: _FBOUNDS = None,
    bin_edges: _BINEDGES = None,
    radius_bounds: _RADBOUNDS = None,
) -> None:
    """Monte Carlo MLE analysis: generate mocks, fit with MLE, store results."""
    from ..mc.mle import MCMLE
    from numcosmo_py import Ncm

    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    warnings.filterwarnings("ignore")

    mset = build_mock_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=true_ra, dec=true_dec, z=redshift, cDelta=cDelta, log10MDelta=log10m,
    )
    generator = build_generator(
        mset, clusters_path=clusters_path,
        true_ra=true_ra, true_dec=true_dec,
        r_min=r_min, r_max=r_max, r_miscenter=r_miscenter,
        n_gals=n_gals, p_cut=p_cut, delta_z=delta_z, z_max=z_max,
        coord_system=coord_system,
    )
    fit_mset = build_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=true_ra, dec=true_dec, z=redshift, cDelta=cDelta, log10MDelta=log10m,
    )
    fparams_list = parse_fparams(fparams)
    bounds_list = parse_fparams_bounds(fparams_bounds)
    bounds_dict = dict(zip(fparams_list, bounds_list)) if bounds_list else None
    lik_factory = build_likelihood_factory(
        fit_mset,
        likelihood_type=likelihood,
        coord_system=coord_system,
        fparams=fparams_list,
        fparams_bounds=bounds_list,
        bin_edges=parse_bin_edges(bin_edges),
        radius_bounds=parse_radius_bounds(radius_bounds),
    )
    true_params = {"NcHaloMassSummary:log10MDelta": log10m}

    mc = MCMLE(
        generator=generator,
        likelihood_factory=lik_factory,
        n_iter=n_iter,
        output=output,
        true_params=true_params,
        seed=seed,
        fparams_bounds=bounds_dict,
    )
    df = mc.run()
    result = mc.summary()

    typer.echo(f"\n✓ Done. {result.n_successful}/{result.n_total} successful ({result.success_rate*100:.1f}%)")
    for param, stats in result.per_param.items():
        typer.echo(f"  {param}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  bias={stats.get('bias', float('nan')):.4f}")


# ---------------------------------------------------------------------------
# mc-posterior subcommand
# ---------------------------------------------------------------------------

@app.command("mc-posterior")
def mc_posterior(
    output: _OUTPUT,
    n_iter: _NITER,
    true_ra: _RA,
    true_dec: _DEC,
    log10m: _LOG10M,
    redshift: _REDSHIFT,
    r_min: _RMIN,
    r_max: _RMAX,
    r_miscenter: _RMIS,
    n_gals: _NGALS,
    clusters_path: _CLUSTERS,
    nsamples: Annotated[int, typer.Option("--nsamples", help="MCMC samples per iteration.")],
    seed: _SEED = None,
    nwalkers: Annotated[int, typer.Option("--nwalkers", help="MCMC walkers.")] = 32,
    nthreads: Annotated[int, typer.Option("--nthreads", help="MCMC threads.")] = 1,
    burn_in: Annotated[int, typer.Option("--burn-in", help="Burn-in samples to discard.")] = 0,
    progress: Annotated[bool, typer.Option("--progress/--no-progress", help="Show MCMC progress.")] = False,
    H0: _H0 = 70.0,
    OmegaB: _OmB = 0.045,
    OmegaC: _OmC = 0.255,
    w: _W = -1.0,
    Omegak: _OmK = 0.0,
    cDelta: _CDELTA = 4.0,
    mass_def: _MASSDEF = MassDefinition.CRITICAL,
    mass_delta: _MASSDELTA = 200.0,
    density_profile: _PROFILE = DensityProfile.NFW,
    coord_system: _COORD = CoordSystem.CELESTIAL,
    p_cut: _PCUT = 0.98,
    delta_z: _DZ = 0.2,
    z_max: _ZMAX = 2.5,
    likelihood: _LIK = LikelihoodType.BINNED_SHEAR,
    fparams: _FPARAMS = None,
    fparams_bounds: _FBOUNDS = None,
    bin_edges: _BINEDGES = None,
    radius_bounds: _RADBOUNDS = None,
) -> None:
    """Monte Carlo posterior analysis: generate mocks, run MCMC, store summary stats."""
    from ..mc.posterior import MCPosterior
    from numcosmo_py import Ncm

    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    warnings.filterwarnings("ignore")

    mset = build_mock_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=true_ra, dec=true_dec, z=redshift, cDelta=cDelta, log10MDelta=log10m,
    )
    generator = build_generator(
        mset, clusters_path=clusters_path,
        true_ra=true_ra, true_dec=true_dec,
        r_min=r_min, r_max=r_max, r_miscenter=r_miscenter,
        n_gals=n_gals, p_cut=p_cut, delta_z=delta_z, z_max=z_max,
        coord_system=coord_system,
    )
    fit_mset = build_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=true_ra, dec=true_dec, z=redshift, cDelta=cDelta, log10MDelta=log10m,
    )
    fparams_list = parse_fparams(fparams)
    bounds_list = parse_fparams_bounds(fparams_bounds)
    bounds_dict = dict(zip(fparams_list, bounds_list)) if bounds_list else None
    lik_factory = build_likelihood_factory(
        fit_mset,
        likelihood_type=likelihood,
        coord_system=coord_system,
        fparams=fparams_list,
        fparams_bounds=bounds_list,
        bin_edges=parse_bin_edges(bin_edges),
        radius_bounds=parse_radius_bounds(radius_bounds),
    )
    true_params = {"NcHaloMassSummary:log10MDelta": log10m}

    mc = MCPosterior(
        generator=generator,
        likelihood_factory=lik_factory,
        n_iter=n_iter,
        output=output,
        true_params=true_params,
        seed=seed,
        nsamples=nsamples,
        nwalkers=nwalkers,
        nthreads=nthreads,
        burn_in=burn_in,
        progress=progress,
        fparams_bounds=bounds_dict,
    )
    df = mc.run()
    result = mc.summary()

    typer.echo(f"\n✓ Done. {result.n_successful}/{result.n_total} successful ({result.success_rate*100:.1f}%)")
    for param, stats in result.per_param.items():
        typer.echo(f"  {param}:")
        typer.echo(f"    bias (mean)   = {stats['bias_mean']:.4f}")
        typer.echo(f"    bias (median) = {stats['bias_median']:.4f}")
        typer.echo(f"    uncertainty   = {stats['uncertainty_1sigma']:.4f} (1σ)")
        typer.echo(f"    coverage 68%  = {stats['coverage_1sigma']:.3f}")
        typer.echo(f"    coverage 95%  = {stats['coverage_2sigma']:.3f}")


# ---------------------------------------------------------------------------
# fit-mle subcommand
# ---------------------------------------------------------------------------

@app.command("fit-mle")
def fit_mle(
    input: Annotated[Path, typer.Option("--input", help="Input parquet file with obs DataFrame.")],
    H0: _H0 = 70.0,
    OmegaB: _OmB = 0.045,
    OmegaC: _OmC = 0.255,
    w: _W = -1.0,
    Omegak: _OmK = 0.0,
    ra: Annotated[float, typer.Option("--ra", help="Halo RA (degrees).")] = 0.0,
    dec: Annotated[float, typer.Option("--dec", help="Halo Dec (degrees).")] = 0.0,
    z: Annotated[float, typer.Option("--z", help="Halo redshift.")] = 0.3,
    cDelta: _CDELTA = 4.0,
    log10MDelta: Annotated[float, typer.Option("--log10MDelta", help="Initial log10(M).")] = 14.5,
    mass_def: _MASSDEF = MassDefinition.CRITICAL,
    mass_delta: _MASSDELTA = 200.0,
    density_profile: _PROFILE = DensityProfile.NFW,
    coord_system: _COORD = CoordSystem.CELESTIAL,
    likelihood: _LIK = LikelihoodType.BINNED_SHEAR,
    fparams: _FPARAMS = None,
    fparams_bounds: _FBOUNDS = None,
    bin_edges: _BINEDGES = None,
    radius_bounds: _RADBOUNDS = None,
    output: Annotated[Optional[Path], typer.Option("--output", help="Optional output parquet for result.")] = None,
) -> None:
    """Run MLE directly on a parquet observation DataFrame."""
    from numcosmo_py import Ncm

    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    warnings.filterwarnings("ignore")

    obs = _load_obs(input)
    mset = build_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=ra, dec=dec, z=z, cDelta=cDelta, log10MDelta=log10MDelta,
    )
    lik_factory = build_likelihood_factory(
        mset,
        likelihood_type=likelihood,
        coord_system=coord_system,
        fparams=parse_fparams(fparams),
        fparams_bounds=parse_fparams_bounds(fparams_bounds),
        bin_edges=parse_bin_edges(bin_edges),
        radius_bounds=parse_radius_bounds(radius_bounds),
    )
    lik = lik_factory(obs)
    lik.prepare_data()
    theta = lik.maximum_likelihood_estimate()
    param_names = [p.split(":", 1)[1] for p in parse_fparams(fparams)]

    typer.echo("MLE result:")
    result_dict = {}
    for name, val in zip(param_names, theta):
        typer.echo(f"  {name} = {val:.6f}")
        result_dict[name] = [val]

    if output is not None:
        import pandas as pd
        pd.DataFrame(result_dict).to_parquet(output, index=False)
        typer.echo(f"Result written to {output}")


# ---------------------------------------------------------------------------
# fit-posterior subcommand
# ---------------------------------------------------------------------------

@app.command("fit-posterior")
def fit_posterior(
    input: Annotated[Path, typer.Option("--input", help="Input parquet file with obs DataFrame.")],
    nsamples: Annotated[int, typer.Option("--nsamples", help="MCMC samples.")],
    H0: _H0 = 70.0,
    OmegaB: _OmB = 0.045,
    OmegaC: _OmC = 0.255,
    w: _W = -1.0,
    Omegak: _OmK = 0.0,
    ra: Annotated[float, typer.Option("--ra", help="Halo RA (degrees).")] = 0.0,
    dec: Annotated[float, typer.Option("--dec", help="Halo Dec (degrees).")] = 0.0,
    z: Annotated[float, typer.Option("--z", help="Halo redshift.")] = 0.3,
    cDelta: _CDELTA = 4.0,
    log10MDelta: Annotated[float, typer.Option("--log10MDelta", help="Initial log10(M).")] = 14.5,
    mass_def: _MASSDEF = MassDefinition.CRITICAL,
    mass_delta: _MASSDELTA = 200.0,
    density_profile: _PROFILE = DensityProfile.NFW,
    coord_system: _COORD = CoordSystem.CELESTIAL,
    nwalkers: Annotated[int, typer.Option("--nwalkers", help="MCMC walkers.")] = 32,
    nthreads: Annotated[int, typer.Option("--nthreads", help="MCMC threads.")] = 1,
    burn_in: Annotated[int, typer.Option("--burn-in", help="Burn-in samples to discard.")] = 0,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = False,
    likelihood: _LIK = LikelihoodType.BINNED_SHEAR,
    fparams: _FPARAMS = None,
    fparams_bounds: _FBOUNDS = None,
    bin_edges: _BINEDGES = None,
    radius_bounds: _RADBOUNDS = None,
    output: Annotated[Optional[Path], typer.Option("--output", help="Output parquet file for chain.")] = None,
) -> None:
    """Run posterior sampling directly on a parquet observation DataFrame."""
    import tempfile
    from numcosmo_py import Ncm

    Ncm.cfg_init()
    Ncm.cfg_set_log_handler(lambda msg: None)
    warnings.filterwarnings("ignore")

    obs = _load_obs(input)
    mset = build_mset(
        H0=H0, OmegaB=OmegaB, OmegaC=OmegaC, w=w, Omegak=Omegak,
        mass_def=mass_def, mass_delta=mass_delta, density_profile=density_profile,
        ra=ra, dec=dec, z=z, cDelta=cDelta, log10MDelta=log10MDelta,
    )
    lik_factory = build_likelihood_factory(
        mset,
        likelihood_type=likelihood,
        coord_system=coord_system,
        fparams=parse_fparams(fparams),
        fparams_bounds=parse_fparams_bounds(fparams_bounds),
        bin_edges=parse_bin_edges(bin_edges),
        radius_bounds=parse_radius_bounds(radius_bounds),
    )
    lik = lik_factory(obs)
    lik.prepare_data()

    chain_file = str(output) if output and str(output).endswith(".h5") else None
    with tempfile.TemporaryDirectory() as tmpdir:
        if chain_file is None:
            chain_file = str(Path(tmpdir) / "chain.h5")
        chain = lik.sample_posterior(
            nsamples=nsamples,
            nwalkers=nwalkers,
            nthreads=nthreads,
            progress=progress,
            filename=chain_file,
        )

    chain = chain.iloc[burn_in:]
    typer.echo("Posterior summary:")
    for col in chain.columns:
        typer.echo(f"  {col}: mean={chain[col].mean():.4f}  std={chain[col].std():.4f}  "
                   f"q16={chain[col].quantile(0.16):.4f}  q84={chain[col].quantile(0.84):.4f}")

    if output is not None and not str(output).endswith(".h5"):
        chain.to_parquet(output, index=False)
        typer.echo(f"Chain written to {output}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_obs(path: Path):
    import pandas as pd
    if not path.exists():
        typer.echo(f"Error: input file not found: {path}", err=True)
        raise typer.Exit(1)
    return pd.read_parquet(path)
