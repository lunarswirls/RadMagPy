#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fresnel OC/SC/CPR for magnetic lossy media (from your equations), driven by material ranges
loaded from an external `materials.py`:

    import materials
    materials.MATERIAL_CLASSES

This version sweeps over frequency (f_min–f_max) at a single incidence angle θi, and:

  - Runs over each selected material class.
  - For each material, computes two cases:
      * mu0:  μ forced to μ0 (μr=1, tanδm=0)
      * muvar: μ variable from material ranges
  - Produces a single 4×2 panel figure:
      * Rows: OC, SC, log10(CPR), δ_normal
      * Col 1: μ = μ0, all materials overplotted
      * Col 2: μ variable, all materials overplotted

Core equations:
  ε* = ε' − i ε'',  μ* = μ' − i μ''
  n = sqrt(ε* μ*),  η = sqrt(μ*/ε*)
  Snell: cosθt = sqrt(1 − (n1/n2)^2 sin^2θi)
  Fresnel (impedance): rs, rp
  E_OC = (rs + rp)/2, E_SC = (rs − rp)/2
  OC = |E_OC|^2, SC = |E_SC|^2, CPR = SC/OC
  k = ω sqrt(μ* ε*), α = Im(k), δ = 1/α

Sweep:
  - frequency axis: f ∈ [f_min_GHz, f_max_GHz], nfreq points (linear)
  - fixed incidence angle θi_deg

Outputs
-------
Per material and case:
  - draws_*.csv : sampled material parameters
  - sweep_summary_*.csv : OC/SC/log10CPR/δ_bulk/δ_normal vs frequency (median + CI)

Global:
  - OC_SC_log10CPR_skin_vs_freq_ALLMATERIALS_... .png :
      4×2 panel figure (μ0 vs μvar; all materials).

Usage examples
--------------
List materials:
  python fresnel_mu_sweep_freq_allmaterials.py --list-materials

Run all materials 1–5 GHz, 30° incidence:
  python fresnel_mu_sweep_freq_allmaterials.py \
      --material all --freq-min-GHz 1.0 --freq-max-GHz 5.0 \
      --nfreq 201 --theta-i-deg 30 --outdir out

Run a single material:
  python fresnel_mu_sweep_freq_allmaterials.py \
      --material "Basaltic rock" --freq-min-GHz 1.0 --freq-max-GHz 5.0 \
      --nfreq 201 --theta-i-deg 30 --outdir out

Author: Dany Waller (script adapted)
"""
import argparse
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
    force_mu0: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ferrimagnetic dispersion only if:
      - not forcing μ0, and
      - ferri_model != "none", and
      - par["ferrimag"] is True.
    """
    if force_mu0:
        # For μ0 case, do NOT apply dispersion; keep μr=1, tanδm=0
        return mu_r, tan_m
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
# Summary + plotting (vs frequency)
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


def summarize_vs_freq(freq_GHz: np.ndarray, arr2d: np.ndarray, ci: float, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, f in enumerate(freq_GHz):
        med, lo, hi = bootstrap_ci_median(arr2d[i, :], ci, n_boot, rng)
        rows.append({"freq_GHz": float(f), "median": med, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def plot_4x2_allmaterials(
    freq_GHz: np.ndarray,
    summaries_by_case: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    out_png: Path,
    title: str,
) -> None:
    """
    4 rows × 2 columns:

      Col 0: μ = μ0 (case 'mu0'), all materials overplotted
      Col 1: μ variable (case 'muvar'), all materials overplotted

    Rows:
      0: OC
      1: SC
      2: log10(CPR)
      3: δ_normal (log scale)
    """
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True, constrained_layout=True)

    quantities = ["OC", "SC", "log10CPR", "delta_normal"]
    ylabels = {
        "OC": "OC = |E_OC|²",
        "SC": "SC = |E_SC|²",
        "log10CPR": "log₁₀(CPR)",
        "delta_normal": "Normal penetration depth δₙ (m)",
    }
    cases = ["mu0", "muvar"]
    col_titles = {
        "mu0": r"$\mu = \mu_0$",
        "muvar": r"$\mu$ variable",
    }

    # Plot
    for col_idx, case in enumerate(cases):
        case_dict = summaries_by_case.get(case, {})
        for row_idx, q in enumerate(quantities):
            ax = axes[row_idx, col_idx]
            q_dict = case_dict.get(q, {})

            for material, df in q_dict.items():
                y = df["median"].to_numpy(float)
                y[~np.isfinite(y)] = np.nan
                ax.plot(df["freq_GHz"], y, label=material)

            ax.set_ylabel(ylabels[q])
            ax.grid(True, which="both", alpha=0.3)
            if q == "delta_normal":
                ax.set_yscale("log")

            if row_idx == 0:
                ax.set_title(col_titles[case])

    # X labels only on bottom row
    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel(r"Frequency $f$ (GHz)")

    # Global title
    fig.suptitle(title, fontsize=14)

    # Single legend outside the right column
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            fontsize=8,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Sweep computation (over frequency, single θi, single material+case)
# ---------------------------------------------------------------------
def run_sweep(
    *,
    material_name: str,
    par: Dict,
    freq_Hz: np.ndarray,
    theta_i_deg: float,
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

    # Draw parameters once per case (used at all frequencies)
    draws = draw_material_parameters(rng=rng, par=par, n_samp=n_samp, force_mu0=force_mu0)

    # Medium 1: vacuum
    eps1_abs = EPS0 + 0j
    mu1_abs = MU0 + 0j
    n1, eta1 = n_eta(eps1_abs, mu1_abs)

    # Angle-dependent terms (single incidence angle)
    theta_i_rad = np.deg2rad(float(theta_i_deg))
    cos_i_scalar = float(np.cos(theta_i_rad))

    nfreq = freq_Hz.size
    n_samp_local = draws.shape[0]

    OC = np.empty((nfreq, n_samp_local), dtype=float)
    SC = np.empty_like(OC)
    log10CPR = np.empty_like(OC)
    delta_bulk_arr = np.empty_like(OC)
    delta_norm_arr = np.empty_like(OC)

    # Base μr, tanδm (frequency-independent ranges)
    mu_r0 = draws["mu_r"].to_numpy(float)
    tan_m0 = draws["tan_m"].to_numpy(float)

    # Loop over frequency
    for j, f_Hz in enumerate(freq_Hz):
        # Apply ferrimag dispersion if enabled (per-frequency)
        mu_r_eff, tan_m_eff = apply_ferrimag_dispersion_if_enabled(
            par=par,
            mu_r=mu_r0,
            tan_m=tan_m0,
            freq_Hz=float(f_Hz),
            ferri_model=ferri_model,
            mu_r_inf=mu_r_inf,
            tau_s=tau_s,
            loss_scale=loss_scale,
            force_mu0=force_mu0,
        )

        # Build ε*, μ* at this f
        eps2_abs = eps_star_abs(
            draws["eps_r"].to_numpy(float),
            draws["tan_e"].to_numpy(float),
            draws["sigma_Spm"].to_numpy(float),
            float(f_Hz),
        )
        mu2_abs = mu_star_abs_base(mu_r_eff, tan_m_eff)

        # Propagation constants
        w = 2.0 * np.pi * float(f_Hz)
        k = w * csqrt(eps2_abs * mu2_abs)  # complex [1/m]

        alpha_bulk = np.abs(np.imag(k))
        alpha_bulk = np.maximum(alpha_bulk, 1e-30)
        delta_bulk = 1.0 / alpha_bulk

        # Medium 2 refractive index / impedance
        n2, eta2 = n_eta(eps2_abs, mu2_abs)

        # Fresnel at this angle and frequency
        ci_vec = np.full(n_samp_local, cos_i_scalar, dtype=float)
        rs, rp, ct = fresnel_rs_rp(eta1, eta2, n1, n2, ci_vec)

        oc, sc, _, logcpr = circular_OC_SC_CPR(rs, rp, floor=floor)
        OC[j, :] = oc
        SC[j, :] = sc
        log10CPR[j, :] = logcpr

        # Bulk penetration depth (independent of angle)
        delta_bulk_arr[j, :] = delta_bulk

        # Normal-to-interface penetration depth (angle-dependent via ct)
        kz = k * ct
        alpha_n = np.abs(np.imag(kz))
        alpha_n = np.maximum(alpha_n, 1e-30)
        delta_n = 1.0 / alpha_n
        delta_norm_arr[j, :] = delta_n

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
            "Material-based Fresnel OC/SC/log10(CPR) sweep vs frequency with optional ferrimag μ dispersion.\n"
            "Loops over materials and plots all materials in a 4×2 figure: μ=μ0 vs μ variable."
        )
    )
    p.add_argument("--list-materials", action="store_true", help="List available material keys and exit.")
    p.add_argument(
        "--material",
        type=str,
        default="all",
        help="Material name (exact key) or 'all' to run over every material class.",
    )

    # Frequency sweep
    p.add_argument("--freq-min-GHz", type=float, default=1.0, help="Minimum frequency [GHz].")
    p.add_argument("--freq-max-GHz", type=float, default=5.0, help="Maximum frequency [GHz].")
    p.add_argument("--nfreq", type=int, default=201, help="Number of frequency samples (linear spacing).")

    # Single incidence angle
    p.add_argument("--theta-i-deg", type=float, default=0.0, help="Incidence angle θi [deg] (0 = normal incidence).")

    p.add_argument("--n_samp", type=int, default=2000, help="Monte-Carlo samples from material ranges.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")

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
    p.add_argument("--ci", type=float, default=0.95, help="Median CI level (0 disables CI calculation).")
    p.add_argument("--n-boot", type=int, default=400, help="Bootstrap replicates per frequency.")
    p.add_argument("--floor", type=float, default=1e-30, help="CPR floor in CPR = (SC+floor)/(OC+floor).")

    # outputs
    p.add_argument("--outdir", type=str, default="mu_fresnel_freq_allmaterials_out", help="Output directory.")
    p.add_argument("--tag", type=str, default="", help="Optional filename tag.")
    return p.parse_args()


def main() -> None:
    args = build_args()

    mats = materials.MATERIAL_CLASSES

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

    # Frequency grid [Hz]
    fmin_GHz = float(args.freq_min_GHz)
    fmax_GHz = float(args.freq_max_GHz)
    nfreq = int(args.nfreq)
    freq_GHz = np.linspace(fmin_GHz, fmax_GHz, nfreq)
    freq_Hz = freq_GHz * 1e9

    # Data structure: summaries_by_case[case]["OC"][material] -> df
    summaries_by_case: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {
        "mu0": {"OC": {}, "SC": {}, "log10CPR": {}, "delta_bulk": {}, "delta_normal": {}},
        "muvar": {"OC": {}, "SC": {}, "log10CPR": {}, "delta_bulk": {}, "delta_normal": {}},
    }

    for material_name in material_list:
        safe_mat = material_name.replace(" ", "_").replace("/", "_")

        par = mats[material_name]

        for case_name, force_mu0 in [("mu0", True), ("muvar", False)]:
            draws, arrays = run_sweep(
                material_name=material_name,
                par=par,
                freq_Hz=freq_Hz,
                theta_i_deg=float(args.theta_i_deg),
                n_samp=int(args.n_samp),
                seed=int(args.seed),
                force_mu0=force_mu0,
                floor=float(args.floor),
                ferri_model=str(args.ferri_model),
                mu_r_inf=float(args.mu_r_inf),
                tau_s=float(args.tau_s),
                loss_scale=float(args.ferri_loss_scale),
            )

            # Save draws (one row per sample; base parameters only)
            draws2 = draws.copy()
            draws2.insert(0, "material", material_name)
            draws2.insert(1, "case", case_name)

            draws_csv = outdir / (
                f"draws_{safe_mat}_{case_name}_"
                f"f{fmin_GHz:g}-{fmax_GHz:g}GHz.csv"
            )

            draws2.to_csv(draws_csv, index=False)
            print("Saved:", draws_csv)

            # Summaries vs frequency
            for k in ["OC", "SC", "log10CPR", "delta_bulk", "delta_normal"]:
                summaries_by_case[case_name][k][material_name] = summarize_vs_freq(
                    freq_GHz=freq_GHz,
                    arr2d=arrays[k],
                    ci=float(args.ci),
                    n_boot=int(args.n_boot),
                    seed=int(args.seed),
                )

            # Save summary CSV for this material+case
            s = summaries_by_case[case_name]
            sum_df = pd.DataFrame({
                "freq_GHz": freq_GHz,

                "OC_median": s["OC"][material_name]["median"],
                "OC_ci_lo": s["OC"][material_name]["ci_lo"],
                "OC_ci_hi": s["OC"][material_name]["ci_hi"],

                "SC_median": s["SC"][material_name]["median"],
                "SC_ci_lo": s["SC"][material_name]["ci_lo"],
                "SC_ci_hi": s["SC"][material_name]["ci_hi"],

                "log10CPR_median": s["log10CPR"][material_name]["median"],
                "log10CPR_ci_lo": s["log10CPR"][material_name]["ci_lo"],
                "log10CPR_ci_hi": s["log10CPR"][material_name]["ci_hi"],

                "delta_bulk_median_m": s["delta_bulk"][material_name]["median"],
                "delta_bulk_ci_lo_m": s["delta_bulk"][material_name]["ci_lo"],
                "delta_bulk_ci_hi_m": s["delta_bulk"][material_name]["ci_hi"],

                "delta_normal_median_m": s["delta_normal"][material_name]["median"],
                "delta_normal_ci_lo_m": s["delta_normal"][material_name]["ci_lo"],
                "delta_normal_ci_hi_m": s["delta_normal"][material_name]["ci_hi"],
            })

            sum_df.insert(0, "material", material_name)
            sum_df.insert(1, "case", case_name)
            sum_df.insert(2, "theta_i_deg", float(args.theta_i_deg))
            sum_df.insert(3, "ferri_model", str(args.ferri_model))

            sum_csv = outdir / (
                f"sweep_summary_{safe_mat}_{case_name}_"
                f"theta{args.theta_i_deg:g}deg_f{fmin_GHz:g}-{fmax_GHz:g}GHz.csv"
            )
            sum_df.to_csv(sum_csv, index=False)
            print("Saved:", sum_csv)

    # Global plot over all materials
    tag = f"_{args.tag}" if args.tag else ""
    ferri_tag = f"_ferri-{args.ferri_model}" if args.ferri_model != "none" else ""
    mat_tag = "ALLMATERIALS" if args.material.lower() == "all" else args.material.replace(" ", "_").replace("/", "_")

    out_png = outdir / (
        f"OC_SC_log10CPR_skin_vs_freq_{mat_tag}_"
        f"theta{args.theta_i_deg:g}deg_f{fmin_GHz:g}-{fmax_GHz:g}GHz{ferri_tag}{tag}.png"
    )
    title = (
        f"Materials: {mat_tag} | θi={args.theta_i_deg:g}° | "
        f"f={fmin_GHz:g}–{fmax_GHz:g} GHz | n_samp={args.n_samp} | ferri={args.ferri_model}"
    )
    plot_4x2_allmaterials(freq_GHz, summaries_by_case, out_png, title)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
