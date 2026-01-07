#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import argparse
import os
import numpy as np
import pandas as pd
import materials
import physics
import roughness_models as rm
import utilities as utils
import plot_util as plotutil

desc="""
Monte-Carlo μ–ε tradeoffs in CPR/OC/SC with frequency dependence.

Features
- Material classes spanning planetary surfaces (basalt, anorthosite, regolith, ice, salts,
  metal-rich, ferrimagnetic soils).
- Frequency sweep with simple dispersive proxies for ε*(f), μ*(f).
- Fresnel with magnetic media, RHCP incidence -> OC/SC coefficients -> CPR.
- Pluggable roughness models: none (default), facet, simple, iem-lite, iem, hagfors
- Stability filter for CPR (avoid OC≈0 blow-ups) plus capping for visualization.
- CSV outputs (raw + cleaned) and summary plots.

Notes
- Dispersion terms are proxies, how to get lab-fit curves?

Run
    python roughness_model_test.py --outdir results

Author: Dany Waller
"""


def build_arguments():
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--n_per_class", type=int, default=2000, help="Samples per material class.")
    p.add_argument("--fmin", type=float, default=0.05, help="Min frequency (GHz).")
    p.add_argument("--fmax", type=float, default=10.0, help="Max frequency (GHz).")
    p.add_argument("--nf", type=int, default=750, help="Number of frequency samples.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--oc_min", type=float, default=1e-6, help="OC floor for stable CPR.")
    p.add_argument("--cpr_cap", type=float, default=100.0, help="Visualization cap for CPR (<=0 disables capping).")
    p.add_argument("--outdir", type=str, default=".", help="Output directory for CSVs and plots.")
    p.add_argument("--save_raw", action="store_true", help="Writes raw data CSV before filtering for stable CPR.")
    p.add_argument("--no_plots", action="store_true", help="Skip plotting (still writes CSVs).")

    # Magnetic toggle
    p.add_argument(
        "--use-mu0",
        action="store_true",
        help="Force μ=μ0 (μr=1, tanδm=0, ferrimag disabled) for all materials at runtime.",
    )

    # Roughness/depol model
    p.add_argument("--roughness-model", type=str, default="none",
                   choices=["simple", "facet", "iem", "none"],
                   help="Roughness/depole model.")

    # IEM-lite options
    p.add_argument("--sigma-h-m", type=float, default=0.01,
                   help="IEM-lite RMS height σh in meters (used if not randomized).")
    p.add_argument("--dihedral-k", type=float, default=0.15,
                   help="IEM-lite dihedral fraction (0–0.5 is reasonable).")
    p.add_argument("--use-random-sigma-h", action="store_true",
                   help="Randomize σh per sample in IEM-lite.")
    p.add_argument("--sigma-h-range", type=float, nargs=2, default=[0.003, 0.03],
                   help="Range for randomized σh (meters) in IEM-lite.")

    # IEM options
    p.add_argument("--iem-psd", type=str, default="gaussian",
                   choices=["gaussian", "exponential", "fractal"],
                   help="Surface PSD shape for IEM.")
    p.add_argument("--iem-shadowing", action="store_true",
                   help="Enable empirical shadowing factor in IEM.")
    p.add_argument("--iem-shadow-m", type=float, default=2.0,
                   help="Shadowing steepness parameter m (larger => stronger shadowing).")
    p.add_argument("--iem-cal-kx", type=float, default=1.0,
                   help="Calibration scalar for IEM spectral term.")
    p.add_argument("--fractal_H", type=float, default=0.7,
                   help="Hurst exponent controlling scale dependence of fractal roughness (D = 3 − H).")
    p.add_argument("--fractal_L_outer", type=float, default=None,
                   help="Outer (largest) roughness scale in meters; sets minimum spatial frequency (q_min).")
    p.add_argument("--fractal_L_inner", type=float, default=0.01,
                   help="Inner (smallest) roughness scale in meters; sets high-frequency roll-off (q_max).")

    # Hagfors options
    p.add_argument("--hag-C", type=float, default=0.3,
                   help="Hagfors shape parameter C (controls angular roll-off).")
    p.add_argument("--hag-n", type=float, default=3.0,
                   help="Hagfors exponent n on cos(theta).")
    p.add_argument("--hag-rho0", type=float, default=0.2,
                   help="Hagfors amplitude scaling rho0.")
    p.add_argument("--hag-pol-mix", type=float, default=0.08,
                   help="Fraction of Hagfors power routed to SC (0..1).")
    return p.parse_args()


def resolve_mu_sampling_and_dispersion(
    *,
    use_mu0: bool,
    rng: np.random.Generator,
    par: dict,
    n: int,
):
    """
    Central switch for magnetic behavior.

    If use_mu0=True:
      - μr is forced to 1
      - tanδm is forced to 0
      - ferrimag is disabled
    Otherwise:
      - sample μr and tanδm from the material class ranges
      - ferrimag is taken from the material class flag
    """
    if use_mu0:
        mu_r0 = np.ones(n, dtype=float)
        tan_m0 = np.zeros(n, dtype=float)
        ferri = False
        return mu_r0, tan_m0, ferri

    mu_r0 = rng.uniform(*par["mu_real"], size=n)

    # Avoid log10(0) if a class ever supplies 0
    tanm_lo = max(par["tan_m"][0], 1e-30)
    tanm_hi = max(par["tan_m"][1], tanm_lo)
    tan_m0 = 10 ** rng.uniform(np.log10(tanm_lo), np.log10(tanm_hi), size=n)

    ferri = bool(par.get("ferrimag", False))
    return mu_r0, tan_m0, ferri


def main():
    args = build_arguments()

    # Correct string comparison (Python uses !=, not "is not")
    if args.roughness_model != "none":
        outdir = args.outdir
        full_outdir = os.path.join(outdir, args.roughness_model)
    else:
        full_outdir = args.outdir

    os.makedirs(full_outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Frequencies
    freqs = np.linspace(args.fmin * 1e9, args.fmax * 1e9, args.nf)

    # Angle & roughness distributions (shared per class)
    theta_deg = np.clip(rng.normal(35.0, 12.0, size=args.n_per_class), 0.0, 80.0)
    theta_rad = np.deg2rad(theta_deg)
    cos_theta = np.cos(theta_rad)
    rms_slope = np.clip(rng.normal(0.25, 0.12, size=args.n_per_class), 0.0, 0.8)

    if args.roughness_model == "none":
        theta_rad = np.zeros_like(theta_rad)
        cos_theta = np.ones_like(cos_theta)

    # Medium 1 (vacuum, relative units)
    eps1 = 1.0 + 0j
    mu1  = 1.0 + 0j
    eta1 = np.sqrt(mu1 / eps1)
    n1   = np.sqrt(mu1 * eps1)

    records = []

    # --- Sample and simulate ---
    for cls_name, par in materials.MATERIAL_CLASSES.items():
        # Draw base params
        eps_r0 = rng.uniform(*par["eps_real"], size=args.n_per_class)
        tan_e0 = 10 ** rng.uniform(
            np.log10(par["tan_e"][0]),
            np.log10(par["tan_e"][1]),
            size=args.n_per_class
        )

        # Avoid log10(0)
        sigma_lo = max(par["sigma"][0], 1e-30)
        sigma_hi = max(par["sigma"][1], sigma_lo)
        sigma0 = 10 ** rng.uniform(np.log10(sigma_lo), np.log10(sigma_hi), size=args.n_per_class)

        # --- Magnetic toggle: decide how to sample μ and whether ferrimag is allowed ---
        mu_r0, tan_m0, ferri = resolve_mu_sampling_and_dispersion(
            use_mu0=args.use_mu0,
            rng=rng,
            par=par,
            n=args.n_per_class,
        )

        for f in freqs:
            # Dispersive ε*(f) and μ*(f)
            eps2 = physics.eps_complex_dispersion(eps_r0, tan_e0, sigma0, f)

            if args.use_mu0:
                # Force μr=1 exactly, independent of f
                mu2 = (1.0 + 0j) * np.ones_like(eps2, dtype=np.complex128)
            else:
                mu2 = physics.mu_complex_dispersion(mu_r0, tan_m0, f, ferri=ferri, rng=rng)

            eta2 = np.sqrt(mu2 / eps2)
            n2   = np.sqrt(mu2 * eps2)

            rs, rp = utils.fresnel_coeffs_magnetic(eta1, eta2, n1, n2, cos_theta)
            OC, SC = utils.circular_components_from_rs_rp(rs, rp)

            # --- Apply chosen roughness model (frequency-aware if needed) ---
            OCr, SCr = rm.roughness_apply(
                OC, SC, args.roughness_model,
                theta_rad=theta_rad,
                rms_slope=rms_slope,
                freq_Hz=f,
                sigma_h_m=args.sigma_h_m,
                dihedral_k=args.dihedral_k,
                rng=rng,
                use_random_sigma_h=args.use_random_sigma_h,
                sigma_h_range=tuple(args.sigma_h_range),
                eps2=eps2, mu2=mu2,  # keep passing mu2; it will be unity when use_mu0=True
                corr_L=0.03,  # meters; adjust or expose as CLI
                iem_psd=args.iem_psd,
                iem_shadow=args.iem_shadowing,
                iem_shadow_m=args.iem_shadow_m,
                iem_cal_kx=args.iem_cal_kx,
                fractal_H=args.fractal_H,
                fractal_L_outer=args.fractal_L_outer,
                fractal_L_inner=args.fractal_L_inner,
                hag_C=args.hag_C,
                hag_n=args.hag_n,
                hag_rho0=args.hag_rho0,
                hag_pol_mix=args.hag_pol_mix
            )

            CPR = np.divide(SCr, OCr, out=np.full_like(SCr, np.nan), where=OCr > 1e-18)

            block = pd.DataFrame({
                "class": cls_name,
                "freq_Hz": f,
                "theta_deg": theta_deg,
                "rms_slope": rms_slope,
                "eps_real0": eps_r0,
                "tan_delta_e0": tan_e0,
                "sigma_Spm": sigma0,
                "mu_real0": mu_r0,      # will be 1.0 if --use-mu0
                "tan_delta_m0": tan_m0, # will be 0.0 if --use-mu0
                "eps2_real": np.real(eps2),
                "eps2_imag": -np.imag(eps2),
                "mu2_real": np.real(mu2),
                "mu2_imag": -np.imag(mu2),
                "OC": OCr,
                "SC": SCr,
                "CPR": CPR
            })
            records.append(block)

    df = pd.concat(records, ignore_index=True)

    if args.roughness_model == "none":
        args.roughness_model = "No Roughness Model"
    elif args.roughness_model == "simple":
        args.roughness_model = "Simple Roughness Model"
    elif args.roughness_model == "facet":
        args.roughness_model = "Facet-only Roughness Model"
    elif args.roughness_model == "iem-lite":
        args.roughness_model = "IEM-Lite Roughness Model"
    elif args.roughness_model == "iem":
        args.roughness_model = "IEM"
    elif args.roughness_model == "hagfors":
        args.roughness_model = "Hagfors Roughness Model"

    # Optionally label output with mu0 toggle
    mu_tag = "mu0" if args.use_mu0 else "muvar"

    if args.save_raw:
        raw_csv = os.path.join(
            args.outdir,
            f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_{mu_tag}_RAW.csv"
        )
        df.to_csv(raw_csv, index=False)
        print(f"Wrote raw samples: {raw_csv}  (rows={len(df):,})")

    # Stable CPR (avoid OC→0 blow-ups)
    df["CPR_stable"] = np.where(df["OC"] >= args.oc_min, df["CPR"], np.nan)
    if args.cpr_cap and args.cpr_cap > 0:
        df["CPR_stable"] = np.where(df["CPR_stable"] <= args.cpr_cap, df["CPR_stable"], args.cpr_cap)

    clean_csv = os.path.join(
        args.outdir,
        f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_{mu_tag}_CLEAN.csv"
    )
    df.to_csv(clean_csv, index=False)
    print(f"Wrote cleaned samples: {clean_csv}")

    if args.no_plots:
        return

    png1 = plotutil.make_freq_summary_figure(
        df,
        outdir=args.outdir,
        roughness_model=args.roughness_model,
        cpr_cap=args.cpr_cap,
        ci=0.95,
        ci_method="bootstrap",
        n_boot=800,
        rng_seed=args.seed,
        show_ci=True,
        legend_ncol=4,
    )
    print("Saved:", png1)

    png2 = plotutil.make_cpr_heatmaps(
        df,
        outdir=args.outdir,
        roughness_model=args.roughness_model,
        target_freqs_GHz=(0.85, 2.37, 7.14),
        nbins=28,
        cpr_cap=args.cpr_cap,
    )
    print("Saved:", png2)

    png3 = plotutil.make_cpr_vs_mu_scatter(
        df,
        outdir=args.outdir,
        roughness_model=args.roughness_model,
        target_freq_GHz=2.38,
    )
    print("Saved:", png3)


if __name__ == "__main__":
    main()
