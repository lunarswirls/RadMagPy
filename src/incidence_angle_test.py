#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresnel OC/SC/CPR for magnetic lossy media (from your equations), driven by material ranges
loaded from an external `materials.py`:

    import materials
    materials.MATERIAL_CLASSES
    materials.MATERIAL_CLASSES_MU0

Implements enhancements requested:
  (2) Plot and summarize log10(CPR) instead of CPR (robust to extreme ratios).
  (4) Optional ferrimagnetic dispersion model for μ*(f) when the material has ferrimag=True.

Core equations:
  ε* = ε' − i ε'',  μ* = μ' − i μ''
  n = sqrt(ε* μ*),  η = sqrt(μ*/ε*)
  Snell: cosθt = sqrt(1 − (n1/n2)^2 sin^2θi)
  Fresnel (impedance): rs, rp
  E_OC = (rs + rp)/2, E_SC = (rs − rp)/2
  OC = |E_OC|^2, SC = |E_SC|^2, CPR = SC/OC
  k = ω sqrt(μ* ε*), α = Im(k), δ = 1/α

Outputs (per material):
  - summary CSV per case with median curves vs θi
  - PNG with 4×1 panels: OC, SC, log10(CPR), normal penetration depth (log scale)

New:
  - If --material all, runs over all materials in the chosen material dict.

Usage examples
--------------
List materials:
  python fresnel_mu_sweep_material.py --list-materials

Single run (material ranges, no μ forcing):
  python fresnel_mu_sweep_material.py --material "Basaltic rock" --freq_GHz 2.38 --outdir out

Run all materials:
  python fresnel_mu_sweep_material.py --material all --freq_GHz 2.38 --outdir out

Compare μ-var vs μ0 (same dielectric draws, μ forced in μ0 case) for all materials:
  python fresnel_mu_sweep_material.py --material all --compare-mu0 --freq_GHz 2.38 --outdir out

Enable ferrimagnetic dispersion (only affects materials with ferrimag=True):
  python fresnel_mu_sweep_material.py --material "Ferrimagnetic soil" --freq_GHz 2.38 --ferri-model debye --outdir out

Author: Dany Waller (script generated + extended)
"""
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import materials

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
EPS0 = 8.8541878128e-12  # F/m
MU0  = 4.0e-7 * np.pi     # H/m
C0   = 299_792_458.0      # m/s


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_name(name: str) -> str:
    """
    Sanitise a material name for use in filenames:
    keep letters, digits, underscore, dash, dot; replace everything else with "_".
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


# ---------------------------------------------------------------------
# Complex math + Fresnel (from your equations)
# ---------------------------------------------------------------------
def csqrt(z: np.ndarray) -> np.ndarray:
    return np.sqrt(z + 0j)


def n_eta(eps_abs: np.ndarray, mu_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # n = sqrt(ε* μ*) normalized by vacuum so it's dimensionless
    n = csqrt(eps_abs * mu_abs) / csqrt(EPS0 * MU0)
    eta = csqrt(mu_abs / eps_abs)
    return n, eta


def cos_theta_t(n1: np.ndarray, n2: np.ndarray, cos_theta_i: np.ndarray) -> np.ndarray:
    sin2_i = 1.0 - cos_theta_i**2
    sin2_t = (n1 / n2) ** 2 * sin2_i
    return csqrt(1.0 - sin2_t)


def fresnel_rs_rp(eta1, eta2, n1, n2, cos_theta_i):
    ct = cos_theta_t(n1, n2, cos_theta_i)
    rs = (eta2 * cos_theta_i - eta1 * ct) / (eta2 * cos_theta_i + eta1 * ct)
    rp = (eta1 * cos_theta_i - eta2 * ct) / (eta1 * cos_theta_i + eta2 * ct)
    return rs, rp, ct


def circular_OC_SC_CPR(rs: np.ndarray, rp: np.ndarray, floor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    e_oc = 0.5 * (rs + rp)
    e_sc = 0.5 * (rs - rp)
    OC = np.abs(e_oc) ** 2
    SC = np.abs(e_sc) ** 2
    CPR = (SC + floor) / (OC + floor)
    log10CPR = np.log10(CPR)
    return OC, SC, CPR, log10CPR


def propagation(eps_abs: np.ndarray, mu_abs: np.ndarray, freq_Hz: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = 2.0 * np.pi * freq_Hz
    k = w * csqrt(eps_abs * mu_abs)
    alpha = np.imag(k)
    delta = np.where(alpha > 0, 1.0 / alpha, np.inf)
    return k, alpha, delta


# ---------------------------------------------------------------------
# Loss models for ε* and μ*, including optional ferrimagnetic dispersion
# ---------------------------------------------------------------------
def eps_star_abs(eps_r: np.ndarray, tan_e: np.ndarray, sigma_Spm: np.ndarray, freq_Hz: float) -> np.ndarray:
    """
    ε*_abs = ε0 ε' - i( ε0 ε' tanδe + σ/ω )
    """
    w = 2.0 * np.pi * freq_Hz
    eps_prime = EPS0 * eps_r
    eps_doubleprime = EPS0 * eps_r * tan_e + sigma_Spm / np.maximum(w, 1e-30)
    return eps_prime - 1j * eps_doubleprime


def mu_star_abs_base(mu_r: np.ndarray, tan_m: np.ndarray) -> np.ndarray:
    """
    μ*_abs = μ0 μ' - i μ0 μ' tanδm
    """
    mu_prime = MU0 * mu_r
    mu_doubleprime = MU0 * mu_r * tan_m
    return mu_prime - 1j * mu_doubleprime


def ferrimag_mu_debye(
    mu_r0: np.ndarray,
    tan_m0: np.ndarray,
    freq_Hz: float,
    *,
    mu_r_inf: float,
    tau_s: float,
    loss_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pragmatic ferrimagnetic dispersion proxy (Debye-like):
      μ_r(ω) = μ_inf + (μ0 - μ_inf) / (1 + (ωτ)^2)
    and an additional loss bump peaked near ωτ ~ 1:
      tanδm_eff = tanδm0 + loss_scale * (ωτ) / (1 + (ωτ)^2)

    Returns (mu_r_eff, tan_m_eff).
    """
    w = 2.0 * np.pi * freq_Hz
    x = w * max(float(tau_s), 1e-30)

    mu_r_eff = float(mu_r_inf) + (mu_r0 - float(mu_r_inf)) / (1.0 + x * x)

    tan_bump = float(loss_scale) * (x / (1.0 + x * x))
    tan_m_eff = tan_m0 + tan_bump
    return mu_r_eff, tan_m_eff


def apply_ferrimag_dispersion_if_enabled(
    *,
    par: Dict,
    mu_r: np.ndarray,
    tan_m: np.ndarray,
    freq_Hz: float,
    ferri_model: str,
    mu_r_inf: float,
    tau_s: float,
    loss_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ferrimagnetic dispersion only if:
      - ferri_model != "none"
      - par["ferrimag"] is True
    """
    if ferri_model == "none":
        return mu_r, tan_m
    if not bool(par.get("ferrimag", False)):
        return mu_r, tan_m

    if ferri_model == "debye":
        return ferrimag_mu_debye(mu_r, tan_m, freq_Hz, mu_r_inf=mu_r_inf, tau_s=tau_s, loss_scale=loss_scale)

    raise ValueError("ferri_model must be 'none' or 'debye'")


# ---------------------------------------------------------------------
# Sampling from MATERIAL_CLASSES ranges (log-uniform for tan and sigma)
# ---------------------------------------------------------------------
def sample_from_ranges(
    *,
    rng: np.random.Generator,
    lo: float,
    hi: float,
    size: int,
    log_uniform: bool,
    min_pos: float = 1e-30,
) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    if log_uniform:
        lo2 = max(lo, min_pos)
        hi2 = max(hi, lo2)
        return 10.0 ** rng.uniform(np.log10(lo2), np.log10(hi2), size=size)
    return rng.uniform(lo, hi, size=size)


def draw_material_parameters(
    *,
    rng: np.random.Generator,
    par: Dict,
    n_samp: int,
    force_mu0: bool,
) -> pd.DataFrame:
    eps_r = sample_from_ranges(rng=rng, lo=par["eps_real"][0], hi=par["eps_real"][1], size=n_samp, log_uniform=False)
    tan_e = sample_from_ranges(rng=rng, lo=par["tan_e"][0], hi=par["tan_e"][1], size=n_samp, log_uniform=True)
    sigma = sample_from_ranges(rng=rng, lo=par["sigma"][0], hi=par["sigma"][1], size=n_samp, log_uniform=True)

    if force_mu0:
        mu_r = np.ones(n_samp, dtype=float)
        tan_m = np.zeros(n_samp, dtype=float)
    else:
        mu_r = sample_from_ranges(rng=rng, lo=par["mu_real"][0], hi=par["mu_real"][1], size=n_samp, log_uniform=False)
        tan_m = sample_from_ranges(rng=rng, lo=par["tan_m"][0], hi=par["tan_m"][1], size=n_samp, log_uniform=True)

    return pd.DataFrame({
        "eps_r": eps_r,
        "tan_e": tan_e,
        "sigma_Spm": sigma,
        "mu_r": mu_r,
        "tan_m": tan_m,
    })


# ---------------------------------------------------------------------
# Summary + plotting
# ---------------------------------------------------------------------
def bootstrap_ci_median(x: np.ndarray, ci: float, n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if ci <= 0 or n_boot <= 0 or x.size == 1:
        return med, np.nan, np.nan
    alpha = (1.0 - ci) / 2.0
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.median(x[idx], axis=1)
    lo = float(np.quantile(meds, alpha))
    hi = float(np.quantile(meds, 1.0 - alpha))
    return med, lo, hi


def summarize_vs_theta(theta_i_deg: np.ndarray, arr2d: np.ndarray, ci: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, th in enumerate(theta_i_deg):
        med, lo, hi = bootstrap_ci_median(arr2d[i, :], ci, n_boot, rng)
        rows.append({"theta_i_deg": float(th), "median": med, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def plot_4panel(
    theta_i_deg: np.ndarray,
    summaries: Dict[str, Dict[str, pd.DataFrame]],
    out_png: Path,
    title: str,
) -> None:
    """
    Plot OC, SC, log10(CPR), and ONLY the normal-to-interface penetration depth:

        δ_normal(θ) = 1 / Im( k * cosθ_t )

    Expected keys in `summaries`:
      summaries["OC"][case]           -> df(theta_i_deg, median, ci_lo, ci_hi)
      summaries["SC"][case]           -> ...
      summaries["log10CPR"][case]     -> ...
      summaries["delta_normal"][case] -> ...

    Notes:
      - The last panel plots δ_normal on a log y-scale.
      - Bulk skin depth δ_bulk is intentionally NOT plotted.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True, constrained_layout=True)

    panels = [
        ("OC", "OC = |E_OC|²"),
        ("SC", "SC = |E_SC|²"),
        ("log10CPR", "log₁₀(CPR)"),
    ]

    # First three panels
    for ax, (k, ylab) in zip(axes[:3], panels):
        if k not in summaries:
            raise KeyError(f"plot_4panel expected summaries['{k}'] but it was missing.")
        for case, df in summaries[k].items():
            y = df["median"].to_numpy(float)
            y[~np.isfinite(y)] = np.nan
            ax.plot(df["theta_i_deg"], y, label=case)

            if ("ci_lo" in df.columns) and ("ci_hi" in df.columns):
                lo = df["ci_lo"].to_numpy(float)
                hi = df["ci_hi"].to_numpy(float)
                lo[~np.isfinite(lo)] = np.nan
                hi[~np.isfinite(hi)] = np.nan
                if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                    ax.fill_between(df["theta_i_deg"], lo, hi, alpha=0.15)

        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)

    # Fourth panel: δ_normal only
    ax = axes[3]
    k = "delta_normal"
    if k not in summaries:
        raise KeyError(
            "plot_4panel expected summaries['delta_normal'] but it was missing. "
            "Compute and summarize δ_normal(θ) = 1/Im(k cosθ_t)."
        )

    for case, df in summaries[k].items():
        y = df["median"].to_numpy(float)
        y[~np.isfinite(y)] = np.nan
        ax.plot(df["theta_i_deg"], y, label=f"{case}: δ_normal")

        if ("ci_lo" in df.columns) and ("ci_hi" in df.columns):
            lo = df["ci_lo"].to_numpy(float)
            hi = df["ci_hi"].to_numpy(float)
            lo[~np.isfinite(lo)] = np.nan
            hi[~np.isfinite(hi)] = np.nan
            if np.any(np.isfinite(lo)) and np.any(np.isfinite(hi)):
                ax.fill_between(df["theta_i_deg"], lo, hi, alpha=0.12)

    ax.set_ylabel("Normal penetration depth δₙ (m)")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_yscale("log")

    axes[-1].set_xlabel(r"Incidence angle $\theta_i$ (deg)")
    axes[0].set_title(title)

    # Legend on last axis for clarity
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------
# Sweep computation
# ---------------------------------------------------------------------
def run_sweep(
    *,
    material_name: str,
    par: Dict,
    freq_Hz: float,
    theta_i_deg: np.ndarray,
    n_samp: int,
    seed: int,
    force_mu0: bool,
    floor: float,
    ferri_model: str,
    mu_r_inf: float,
    tau_s: float,
    loss_scale: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)

    # Draw parameters once per case (used at all angles)
    draws = draw_material_parameters(rng=rng, par=par, n_samp=n_samp, force_mu0=force_mu0)

    # Apply ferrimag dispersion if enabled (per-frequency effect)
    mu_r_eff, tan_m_eff = apply_ferrimag_dispersion_if_enabled(
        par=par,
        mu_r=draws["mu_r"].to_numpy(float),
        tan_m=draws["tan_m"].to_numpy(float),
        freq_Hz=freq_Hz,
        ferri_model=ferri_model,
        mu_r_inf=mu_r_inf,
        tau_s=tau_s,
        loss_scale=loss_scale,
    )
    draws["mu_r_eff"] = mu_r_eff
    draws["tan_m_eff"] = tan_m_eff
    draws["ferrimag_applied"] = bool(par.get("ferrimag", False) and ferri_model != "none")

    # Build ε*, μ*
    eps2_abs = eps_star_abs(
        draws["eps_r"].to_numpy(float),
        draws["tan_e"].to_numpy(float),
        draws["sigma_Spm"].to_numpy(float),
        freq_Hz,
    )
    mu2_abs = mu_star_abs_base(draws["mu_r_eff"].to_numpy(float), draws["tan_m_eff"].to_numpy(float))

    w = 2.0 * np.pi * freq_Hz
    k = w * csqrt(eps2_abs * mu2_abs)  # complex [1/m]
    alpha_bulk = np.abs(np.imag(k))  # robust to branch sign
    alpha_bulk = np.maximum(alpha_bulk, 1e-30)
    delta_bulk = 1.0 / alpha_bulk  # [m]

    # Medium 1: vacuum
    eps1_abs = EPS0 + 0j
    mu1_abs = MU0 + 0j
    n1, eta1 = n_eta(eps1_abs, mu1_abs)
    n2, eta2 = n_eta(eps2_abs, mu2_abs)

    # Allocate arrays: shape (n_theta, n_samp)
    ntheta = theta_i_deg.size
    OC = np.empty((ntheta, n_samp), dtype=float)
    SC = np.empty_like(OC)
    log10CPR = np.empty_like(OC)

    theta_i_rad = np.deg2rad(theta_i_deg)
    cos_i = np.cos(theta_i_rad)

    # propagation terms (independent of theta_i)
    _, _, delta = propagation(eps2_abs, mu2_abs, freq_Hz)

    alpha_bulk_arr = np.empty((ntheta, n_samp), dtype=float)
    delta_bulk_arr = np.empty((ntheta, n_samp), dtype=float)
    alpha_norm_arr = np.empty((ntheta, n_samp), dtype=float)
    delta_norm_arr = np.empty((ntheta, n_samp), dtype=float)

    for i, ci in enumerate(cos_i):
        rs, rp, ct = fresnel_rs_rp(eta1, eta2, n1, n2, np.full(n_samp, ci, dtype=float))

        oc, sc, _, logcpr = circular_OC_SC_CPR(rs, rp, floor=floor)
        OC[i, :] = oc
        SC[i, :] = sc
        log10CPR[i, :] = logcpr

        # Bulk is independent of theta, but store as a flat curve for plotting convenience
        alpha_bulk_arr[i, :] = alpha_bulk
        delta_bulk_arr[i, :] = delta_bulk

        # Normal-to-interface attenuation depends on theta through ct
        kz = k * ct
        alpha_n = np.abs(np.imag(kz))
        alpha_n = np.maximum(alpha_n, 1e-30)
        delta_n = 1.0 / alpha_n
        alpha_norm_arr[i, :] = alpha_n
        delta_norm_arr[i, :] = delta_n

    arrays = {
        "OC": OC,
        "SC": SC,
        "log10CPR": log10CPR,
        "delta_bulk": delta_bulk_arr,
        "delta_normal": delta_norm_arr,
    }

    return draws, arrays


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Material-based Fresnel OC/SC/log10(CPR) sweep vs incidence angle θi "
            "with optional ferrimagnetic μ dispersion. "
            "Use --material <name> for a single material or --material all for all materials."
        )
    )
    p.add_argument("--list-materials", action="store_true", help="List available material keys and exit.")
    p.add_argument(
        "--material",
        type=str,
        default="Basaltic rock",
        help="Material name (exact key) or 'all' to run all materials.",
    )

    p.add_argument("--freq_GHz", type=float, default=2.38, help="Frequency [GHz].")
    p.add_argument("--n_samp", type=int, default=2000, help="Monte-Carlo samples from material ranges.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

    p.add_argument("--theta_i_min", type=float, default=0.0, help="Min θi [deg].")
    p.add_argument("--theta_i_max", type=float, default=80.0, help="Max θi [deg].")
    p.add_argument("--ntheta", type=int, default=401, help="Number of θi samples.")

    # μ control
    p.add_argument("--use-mu0", action="store_true", help="Force μ*=μ0 (μr=1, tanδm=0) regardless of material ranges.")
    p.add_argument("--use-mu0-class-dict", action="store_true",
                   help="Use materials.MATERIAL_CLASSES_MU0 for parameter ranges (alternative to --use-mu0).")
    p.add_argument("--compare-mu0", action="store_true",
                   help="Run both μ-var and μ0 cases (overlays); μ0 case forces μ*=μ0.")

    # ferrimagnetic dispersion
    p.add_argument("--ferri-model", type=str, default="none", choices=["none", "debye"],
                   help="Ferrimagnetic μ*(f) proxy model (applies only if material ferrimag=True).")
    p.add_argument("--mu-r-inf", type=float, default=1.0,
                   help="Debye model: μ_r(ω→∞) baseline (>=1 typically).")
    p.add_argument("--tau-s", type=float, default=5e-9,
                   help="Debye model: relaxation time τ [s].")
    p.add_argument("--ferri-loss-scale", type=float, default=0.03,
                   help="Debye model: additive tanδm bump scale near ωτ~1.")

    # stats/plotting
    p.add_argument("--ci", type=float, default=0.95, help="Median CI level (0 disables CI shading).")
    p.add_argument("--n-boot", type=int, default=400, help="Bootstrap replicates per θi.")
    p.add_argument("--floor", type=float, default=1e-30, help="CPR floor in CPR = (SC+floor)/(OC+floor).")

    # outputs
    p.add_argument("--outdir", type=str, default="mu_fresnel_material_out", help="Output directory.")
    p.add_argument("--tag", type=str, default="", help="Optional filename tag.")
    return p.parse_args()


def main() -> None:
    args = build_args()

    # Choose which dict supplies ranges
    mats = materials.MATERIAL_CLASSES_MU0 if args.use_mu0_class_dict else materials.MATERIAL_CLASSES

    if args.list_materials:
        print("Available materials:")
        for k in sorted(mats.keys()):
            print(" -", k)
        return

    # Determine which materials to run
    if args.material.lower() == "all":
        material_list = sorted(mats.keys())
    else:
        if args.material not in mats:
            choices = "\n".join([f"  - {k}" for k in sorted(mats.keys())])
            raise SystemExit(f"Unknown material '{args.material}'. Available:\n{choices}")
        material_list = [args.material]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    freq_Hz = float(args.freq_GHz) * 1e9
    theta_i_deg = np.linspace(float(args.theta_i_min), float(args.theta_i_max), int(args.ntheta))

    # Cases (same for all materials)
    if args.compare_mu0:
        cases = [("muvar", False), ("mu0", True)]
    else:
        force_mu0 = bool(args.use_mu0 or args.use_mu0_class_dict)
        cases = [("mu0" if force_mu0 else "muvar", force_mu0)]

    for material_name in material_list:
        par = mats[material_name]
        mat_safe = safe_name(material_name)

        print(f"=== Running material: {material_name} ===")

        # Fresh summaries per material (for plotting)
        summaries = {"OC": {}, "SC": {}, "log10CPR": {}, "delta_bulk": {}, "delta_normal": {}}

        for case_name, force_mu0 in cases:
            draws, arrays = run_sweep(
                material_name=material_name,
                par=par,
                freq_Hz=freq_Hz,
                theta_i_deg=theta_i_deg,
                n_samp=int(args.n_samp),
                seed=int(args.seed),
                force_mu0=force_mu0,
                floor=float(args.floor),
                ferri_model=str(args.ferri_model),
                mu_r_inf=float(args.mu_r_inf),
                tau_s=float(args.tau_s),
                loss_scale=float(args.ferri_loss_scale),
            )

            # Save draws (one row per sample)
            draws2 = draws.copy()
            draws2.insert(0, "material", material_name)
            draws2.insert(1, "case", case_name)
            draws2.insert(2, "freq_GHz", float(args.freq_GHz))

            draws_csv = outdir / f"draws_{mat_safe}_{case_name}_f{args.freq_GHz:g}GHz.csv"
            draws2.to_csv(draws_csv, index=False)

            # Summaries vs θi
            for k in summaries.keys():
                summaries[k][case_name] = summarize_vs_theta(
                    theta_i_deg=theta_i_deg,
                    arr2d=arrays[k],
                    ci=float(args.ci),
                    n_boot=int(args.n_boot),
                    seed=int(args.seed),
                )

            # Save summary CSV (median + CI) for this material+case
            sum_df = pd.DataFrame({
                "theta_i_deg": theta_i_deg,

                "OC_median": summaries["OC"][case_name]["median"],
                "OC_ci_lo": summaries["OC"][case_name]["ci_lo"],
                "OC_ci_hi": summaries["OC"][case_name]["ci_hi"],

                "SC_median": summaries["SC"][case_name]["median"],
                "SC_ci_lo": summaries["SC"][case_name]["ci_lo"],
                "SC_ci_hi": summaries["SC"][case_name]["ci_hi"],

                "log10CPR_median": summaries["log10CPR"][case_name]["median"],
                "log10CPR_ci_lo": summaries["log10CPR"][case_name]["ci_lo"],
                "log10CPR_ci_hi": summaries["log10CPR"][case_name]["ci_hi"],

                "delta_bulk_median_m": summaries["delta_bulk"][case_name]["median"],
                "delta_bulk_ci_lo_m": summaries["delta_bulk"][case_name]["ci_lo"],
                "delta_bulk_ci_hi_m": summaries["delta_bulk"][case_name]["ci_hi"],

                "delta_normal_median_m": summaries["delta_normal"][case_name]["median"],
                "delta_normal_ci_lo_m": summaries["delta_normal"][case_name]["ci_lo"],
                "delta_normal_ci_hi_m": summaries["delta_normal"][case_name]["ci_hi"],
            })

            sum_df.insert(0, "material", material_name)
            sum_df.insert(1, "case", case_name)
            sum_df.insert(2, "freq_GHz", float(args.freq_GHz))
            sum_df.insert(3, "ferri_model", str(args.ferri_model))

            sum_csv = outdir / f"sweep_summary_{mat_safe}_{case_name}_f{args.freq_GHz:g}GHz.csv"
            sum_df.to_csv(sum_csv, index=False)

            print("Saved:", draws_csv)
            print("Saved:", sum_csv)

        # Plot per material
        tag = f"_{args.tag}" if args.tag else ""
        ferri_tag = f"_ferri-{args.ferri_model}" if args.ferri_model != "none" else ""
        out_png = outdir / f"OC_SC_log10CPR_skin_vs_theta_{mat_safe}_f{args.freq_GHz:g}GHz{ferri_tag}{tag}.png"
        title = f"{material_name} | f={args.freq_GHz:g} GHz | n_samp={args.n_samp} | ferri={args.ferri_model}"
        plot_4panel(theta_i_deg, summaries, out_png, title)
        print("Saved:", out_png)

    print("Done.")


if __name__ == "__main__":
    main()
