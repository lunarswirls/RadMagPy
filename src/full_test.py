#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import materials
import physics
import roughness_models as rm
import utilities as utils

desc="""
Monte-Carlo μ–ε tradeoffs in CPR/OC/SC with frequency dependence.

Features
- Material classes spanning planetary surfaces (basalt, anorthosite, regolith, ice, salts,
  metal-rich, ferrimagnetic soils).
- Frequency sweep with simple dispersive proxies for ε*(f), μ*(f).
- Fresnel with magnetic media, RHCP incidence -> OC/SC coefficients -> CPR.
- Pluggable roughness models: none (default), facet, simple, iem-lite, iem, hagfors
- Stability filter for CPR (avoid OC≈0 blow-ups), optional capping for visualization.
- CSV outputs (raw + cleaned) and several summary plots.

Notes
- Dispersion terms are proxies; swap in lab-fit curves when available.

Run
    python full_test.py --outdir results

Author: Dany Waller
"""


def build_arguments():
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--n_per_class", type=int, default=2000, help="Samples per material class.")
    p.add_argument("--fmin", type=float, default=0.05, help="Min frequency (GHz).")
    p.add_argument("--fmax", type=float, default=20.0, help="Max frequency (GHz).")
    p.add_argument("--nf", type=int, default=100, help="Number of frequency samples.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    p.add_argument("--oc_min", type=float, default=1e-6, help="OC floor for stable CPR.")
    p.add_argument("--cpr_cap", type=float, default=10.0, help="Visualization cap for CPR (<=0 disables capping).")
    p.add_argument("--outdir", type=str, default=".", help="Output directory for CSVs and plots.")
    p.add_argument("--no_plots", action="store_true", help="Skip plotting (still writes CSVs).")

    # Roughness control
    p.add_argument("--roughness-model", type=str, default="none",
                   choices=["simple", "facet", "iem-lite", "iem", "hagfors", "none"],
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
                   choices=["gaussian", "exponential"],
                   help="Surface PSD shape for IEM.")
    p.add_argument("--iem-shadowing", action="store_true",
                   help="Enable empirical shadowing factor in IEM.")
    p.add_argument("--iem-shadow-m", type=float, default=2.0,
                   help="Shadowing steepness parameter m (larger => stronger shadowing).")
    p.add_argument("--iem-cal-kx", type=float, default=1.0,
                   help="Calibration scalar for IEM spectral term.")

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


def main():
    args = build_arguments()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Frequencies
    freqs = np.linspace(args.fmin*1e9, args.fmax*1e9, args.nf)

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
    eta1 = np.sqrt(mu1/eps1)
    n1   = np.sqrt(mu1*eps1)

    records = []

    # --- Sample and simulate ---
    for cls_name, par in materials.MATERIAL_CLASSES.items():
        # Draw base params
        eps_r0 = rng.uniform(*par["eps_real"], size=args.n_per_class)
        tan_e0 = 10**rng.uniform(np.log10(par["tan_e"][0]),
                                 np.log10(par["tan_e"][1]),
                                 size=args.n_per_class)
        # Avoid log10(0)
        sigma_lo = max(par["sigma"][0], 1e-9)
        sigma0 = 10**rng.uniform(np.log10(sigma_lo),
                                 np.log10(par["sigma"][1]),
                                 size=args.n_per_class)
        mu_r0  = rng.uniform(*par["mu_real"], size=args.n_per_class)
        tan_m0 = 10**rng.uniform(np.log10(par["tan_m"][0]),
                                 np.log10(par["tan_m"][1]),
                                 size=args.n_per_class)

        for f in freqs:
            # Dispersive ε*(f) and μ*(f)
            eps2 = physics.eps_complex_dispersion(eps_r0, tan_e0, sigma0, f)
            mu2  = physics.mu_complex_dispersion(mu_r0, tan_m0, f, ferri=par["ferrimag"], rng=rng)

            eta2 = np.sqrt(mu2/eps2)
            n2   = np.sqrt(mu2*eps2)

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
                eps2=eps2, mu2=mu2,
                corr_L=0.03,  # meters; adjust or expose as CLI
                iem_psd=args.iem_psd,
                iem_shadow=args.iem_shadowing,
                iem_shadow_m=args.iem_shadow_m,
                iem_cal_kx=args.iem_cal_kx,
                hag_C=args.hag_C, hag_n=args.hag_n, hag_rho0=args.hag_rho0, hag_pol_mix=args.hag_pol_mix
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
                "mu_real0": mu_r0,
                "tan_delta_m0": tan_m0,
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

    if args.roughness_model == 'none':
        args.roughness_model = 'No Roughness Model'
    elif args.roughness_model == 'simple':
        args.roughness_model = 'Simple Roughness Model'
    elif args.roughness_model == 'facet':
        args.roughness_model = 'Facet-only Roughness Model'
    elif args.roughness_model == 'iem-lite':
        args.roughness_model = 'IEM-Lite Roughness Model'
    elif args.roughness_model == 'iem':
        args.roughness_model = 'IEM'
    elif args.roughness_model == 'hagfors':
        args.roughness_model = 'Hagfors Roughness Model'

    # Save raw
    raw_csv = os.path.join(args.outdir, f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_RAW.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Wrote raw samples: {raw_csv}  (rows={len(df):,})")

    # Stable CPR (avoid OC→0 blow-ups)
    df["CPR_stable"] = np.where(df["OC"] >= args.oc_min, df["CPR"], np.nan)
    if args.cpr_cap and args.cpr_cap > 0:
        df["CPR_stable"] = np.where(df["CPR_stable"] <= args.cpr_cap, df["CPR_stable"], args.cpr_cap)

    clean_csv = os.path.join(args.outdir, f"planetary_mu_epsilon_cpr_{str(args.roughness_model).lower().replace(' ', '_')}_CLEAN.csv")
    df.to_csv(clean_csv, index=False)
    print(f"Wrote cleaned samples: {clean_csv}")

    if args.no_plots:
        return

    # ----------- Plots -----------
    g = df.groupby(["class", "freq_Hz"], as_index=False).agg(
        CPR_med=("CPR_stable", "median"),
        OC_med=("OC", "median"),
        SC_med=("SC", "median"),
        n=("CPR_stable", "count"),
    )

    # 1) Median CPR vs frequency by class
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["CPR_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median CPR (stable)")
    plt.title(f"Median CPR vs Frequency by Material Class\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"CPR_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 2) Median OC vs frequency
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["OC_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median OC")
    plt.title(f"Median OC vs Frequency\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"OC_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 3) Median SC vs frequency
    plt.figure()
    for cls in g["class"].unique():
        d = g[g["class"] == cls]
        plt.plot(d["freq_Hz"]/1e9, d["SC_med"], marker="o", label=cls)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Median SC")
    plt.title(f"Median SC vs Frequency\n{args.roughness_model}")
    plt.legend(fontsize="small", ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"SC_vs_freq_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    # 4) CPR heatmaps across (μ', ε') at ~0.85, 2.37, 7,14 GHz (use nearest frequencies)
    def median_cpr_grid(dfin, f_Hz, nbins=28):
        sub = dfin[np.isclose(dfin["freq_Hz"], f_Hz)]
        if sub.empty:
            return None, None, None
        mu = sub["mu2_real"].to_numpy()
        epsr = sub["eps2_real"].to_numpy()
        cpr = sub["CPR_stable"].to_numpy()
        if len(mu) == 0 or np.all(np.isnan(cpr)):
            return None, None, None
        mu_edges = np.linspace(np.nanmin(mu), np.nanmax(mu), nbins+1)
        eps_edges = np.linspace(np.nanmin(epsr), np.nanmax(epsr), nbins+1)
        grid = np.full((nbins, nbins), np.nan)
        for i in range(nbins):
            for j in range(nbins):
                m = (mu >= mu_edges[i]) & (mu < mu_edges[i+1]) & \
                    (epsr >= eps_edges[j]) & (epsr < eps_edges[j+1])
                if np.any(m):
                    grid[i, j] = np.nanmedian(cpr[m])
        mu_cent = 0.5*(mu_edges[:-1] + mu_edges[1:])
        eps_cent = 0.5*(eps_edges[:-1] + eps_edges[1:])
        return mu_cent, eps_cent, grid

    unique_freqs = np.sort(df["freq_Hz"].unique())
    for fGHz in [0.85, 2.37, 7.14]:
        f_sel = utils.nearest(unique_freqs, fGHz*1e9)
        mu_cent, eps_cent, grid = median_cpr_grid(df, f_sel, nbins=28)
        if grid is None:
            continue
        plt.figure()
        extent = [eps_cent.min(), eps_cent.max(), mu_cent.min(), mu_cent.max()]
        plt.imshow(np.flipud(grid), aspect="auto", extent=extent, origin="lower")
        plt.xlabel("ε' (real permittivity)")
        plt.ylabel("μ' (real permeability)")
        plt.title(f"Median CPR (stable) across (μ', ε') at ~{f_sel/1e9:.2f} GHz\n{args.roughness_model}")
        cbar = plt.colorbar()
        lab = "Median CPR"
        if args.cpr_cap and args.cpr_cap > 0:
            lab += f" (capped at {args.cpr_cap:g})"
        cbar.set_label(lab)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, f"CPR_heatmap_{str(args.roughness_model).lower().replace(' ', '_')}_{f_sel/1e9:.2f}GHz.png"), dpi=180)

    # 5) S-band scatter CPR vs μ' (marker size ~ ε')
    s_freq = utils.nearest(unique_freqs, 2.38e9)
    sband = df[np.isclose(df["freq_Hz"], s_freq)].copy()
    plt.figure()
    ms = 6 + 3*(sband["eps2_real"] - sband["eps2_real"].min()) / \
            (sband["eps2_real"].max() - sband["eps2_real"].min() + 1e-12)
    plt.scatter(sband["mu2_real"], sband["CPR_stable"], s=ms, alpha=0.25)
    plt.xlabel("μ' (~S-band)")
    plt.ylabel("CPR (stable)")
    plt.title(f"CPR vs μ' at ~{s_freq/1e9:.2f} GHz\n{args.roughness_model} (marker size ~ ε')")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"CPR_vs_mu_Sband_{str(args.roughness_model).lower().replace(' ', '_')}.png"), dpi=180)

    print(f"Plots saved to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    main()