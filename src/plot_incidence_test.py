#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recreate multi-material Fresnel incidence-angle plots from precomputed sweep
summary CSVs (θ-sweep version).

This script:
  - Scans a directory for files named like: sweep_summary_*.csv
  - Each CSV is assumed to have at least the columns:

        material, case, freq_GHz, ferri_model,
        theta_i_deg,
        OC_median, OC_ci_lo, OC_ci_hi,
        SC_median, SC_ci_lo, SC_ci_hi,
        log10CPR_median, log10CPR_ci_lo, log10CPR_ci_hi,
        delta_bulk_median_m, delta_bulk_ci_lo_m, delta_bulk_ci_hi_m,
        delta_normal_median_m, delta_normal_ci_lo_m, delta_normal_ci_hi_m

    as produced by the θ-sweep script you used earlier.

  - Groups by case ("mu0", "muvar") and material.
  - Reconstructs a 4×2 plot:

        Rows:  OC, SC, log10(CPR), δ_normal
        Col 0: μ = μ0   (case == "mu0"), all materials overplotted
        Col 1: μ variable (case == "muvar"), all materials overplotted

  - No recomputation of Fresnel quantities; purely post-processing.

Y-limits per row are unified between the two columns using the *union*
of the auto-calculated limits (based on the plotted data).

Usage
-----
Example:

  python replot_mu_fresnel_theta_allmaterials.py \
      --indir /Users/danywaller/Projects/moon/radmag/theta_sweeps \
      --freq-GHz 2.38 \
      --ferri-model debye \
      --outpng mu_fresnel_theta_allmaterials_2p38GHz_debye.png

If you only have one frequency and one ferri model in that directory:

  python replot_mu_fresnel_theta_allmaterials.py \
      --indir /Users/danywaller/Projects/moon/radmag/theta_sweeps
"""
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_4x2_allmaterials_theta(
    theta_i_deg: np.ndarray,
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

    After plotting, y-limits for each row are unified between the two
    columns using the union of the auto-calculated limits.
    """
    fig, axes = plt.subplots(4, 2, figsize=(10, 8), sharex=True, constrained_layout=True)

    quantities = ["OC", "SC", "log10CPR", "delta_normal"]
    ylabels = {
        "OC": "OC = |E_OC|²",
        "SC": "SC = |E_SC|²",
        "log10CPR": "log₁₀(CPR)",
        "delta_normal": "δₙ (m)",
    }
    cases = ["mu0", "muvar"]
    col_titles = {
        "mu0": r"$\mu = \mu_0$",
        "muvar": r"$\mu$ variable",
    }

    # ---- First pass: plot everything with Matplotlib's automatic scaling ----
    for col_idx, case in enumerate(cases):
        case_dict = summaries_by_case.get(case, {})
        for row_idx, q in enumerate(quantities):
            ax = axes[row_idx, col_idx]
            q_dict = case_dict.get(q, {})

            for material, df in q_dict.items():
                y = df["median"].to_numpy(float)
                y[~np.isfinite(y)] = np.nan
                ax.plot(df["theta_i_deg"], y, label=material)

            # y-label only on first (left) column
            if col_idx == 0:
                ax.set_ylabel(ylabels[q], fontsize=12)
            else:
                ax.set_ylabel("")
            ax.grid(True, which="both", alpha=0.3)

            if q == "delta_normal":
                ax.set_yscale("log")

            if row_idx == 0:
                ax.set_title(col_titles[case])

    # X labels only on bottom row
    for col_idx in range(2):
        axes[-1, col_idx].set_xlabel(r"Incidence angle $\theta_i$ (deg)", fontsize=12)

    # ---- Second pass: unify y-limits between the two columns for each row ----
    for row_idx, q in enumerate(quantities):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]

        # Get current auto limits
        y0_min, y0_max = ax_left.get_ylim()
        y1_min, y1_max = ax_right.get_ylim()

        # Union of ranges
        y_min = min(y0_min, y1_min)
        y_max = max(y0_max, y1_max)

        # For log axes, make sure lower bound is positive
        if q == "delta_normal":
            # keep the smaller positive of the two, if any
            y_candidates = [v for v in (y0_min, y1_min) if v > 0]
            if y_candidates:
                y_min = min(y_candidates)
            else:
                y_min = 1e-9  # fallback

        ax_left.set_ylim(y_min, y_max)
        ax_right.set_ylim(y_min, y_max)

    # Global title
    fig.suptitle(title, fontsize=14)

    # align y-axis labels
    fig.align_ylabels(axes[:, 0])  # left column only

    # Legend from right column, first row (muvar)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.005, 0.5),
            borderaxespad=0.0,
            fontsize=10,
        )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Loading + grouping
# ---------------------------------------------------------------------
def load_summaries_theta(
    indir: Path,
    freq_filter: float | None,
    ferri_filter: str | None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Dict[str, pd.DataFrame]]], Dict]:
    """
    Scan `indir` for sweep_summary_*.csv and construct summaries_by_case
    for θ-sweep data.

    Returns:
      theta_i_deg_ref : np.ndarray
      summaries_by_case[case][quantity][material] -> DataFrame
      meta : dict with 'freq_GHz', 'ferri_model', 'materials', 'cases'
    """
    pattern = "sweep_summary_*.csv"
    files = sorted(indir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {indir}")

    theta_ref: np.ndarray | None = None
    freq_set: List[float] = []
    ferri_set: List[str] = []
    materials_set: List[str] = []
    cases_set: List[str] = []

    # summaries_by_case[case][quantity][material] = df
    summaries_by_case: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for f in files:
        df = pd.read_csv(f)

        # Basic columns
        if "case" not in df.columns or "material" not in df.columns:
            # Skip anything that does not look like a sweep summary
            continue

        case = str(df["case"].iloc[0])
        material = str(df["material"].iloc[0])
        freq = float(df.get("freq_GHz", pd.Series([np.nan])).iloc[0])
        ferri_model = str(df.get("ferri_model", pd.Series(["unknown"])).iloc[0])

        # Optional filters
        if freq_filter is not None and not np.isfinite(freq):
            continue
        if freq_filter is not None and abs(freq - freq_filter) > 1e-6:
            continue

        if ferri_filter is not None and ferri_model != ferri_filter:
            continue

        theta = df["theta_i_deg"].to_numpy(float)

        if theta_ref is None:
            theta_ref = theta
        else:
            # Sanity check: same grid (within small tolerance)
            if theta.shape != theta_ref.shape or not np.allclose(theta, theta_ref, rtol=0, atol=1e-9):
                raise ValueError(f"Inconsistent theta_i_deg grid in file {f}")

        freq_set.append(freq)
        ferri_set.append(ferri_model)
        materials_set.append(material)
        cases_set.append(case)

        # Prepare quantity DataFrames with 'theta_i_deg','median','ci_lo','ci_hi'
        def make_qdf(median_col: str, lo_col: str, hi_col: str) -> pd.DataFrame:
            return pd.DataFrame({
                "theta_i_deg": theta,
                "median": df[median_col].to_numpy(float),
                "ci_lo": df[lo_col].to_numpy(float),
                "ci_hi": df[hi_col].to_numpy(float),
            })

        qdfs = {
            "OC": make_qdf("OC_median", "OC_ci_lo", "OC_ci_hi"),
            "SC": make_qdf("SC_median", "SC_ci_lo", "SC_ci_hi"),
            "log10CPR": make_qdf("log10CPR_median", "log10CPR_ci_lo", "log10CPR_ci_hi"),
            "delta_bulk": make_qdf("delta_bulk_median_m", "delta_bulk_ci_lo_m", "delta_bulk_ci_hi_m"),
            "delta_normal": make_qdf("delta_normal_median_m", "delta_normal_ci_lo_m", "delta_normal_ci_hi_m"),
        }

        if case not in summaries_by_case:
            summaries_by_case[case] = {q: {} for q in qdfs.keys()}

        for q, qdf in qdfs.items():
            summaries_by_case[case][q][material] = qdf

    if theta_ref is None:
        raise RuntimeError("No valid summary CSVs were loaded.")

    # Collapse meta info
    freq_unique = sorted({round(f, 6) for f in freq_set if np.isfinite(f)})
    ferri_unique = sorted(set(ferri_set))
    materials_unique = sorted(set(materials_set))
    cases_unique = sorted(set(cases_set))

    meta = {
        "freq_GHz": freq_unique,
        "ferri_model": ferri_unique,
        "materials": materials_unique,
        "cases": cases_unique,
    }

    return theta_ref, summaries_by_case, meta


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Recreate μ0 vs μ-variable multi-material Fresnel incidence-angle plots "
            "from existing sweep_summary_*.csv files (θ-sweep), without recomputing sweeps."
        )
    )
    p.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Directory containing sweep_summary_*.csv files produced by the θ-sweep script.",
    )
    p.add_argument(
        "--outpng",
        type=str,
        default="OC_SC_log10CPR_skin_vs_theta_replot.png",
        help="Output PNG filename.",
    )
    p.add_argument(
        "--freq-GHz",
        type=float,
        default=None,
        help="Optional filter on freq_GHz [GHz]. If set, only CSVs with this frequency are used.",
    )
    p.add_argument(
        "--ferri-model",
        type=str,
        default=None,
        help="Optional filter on ferri_model (e.g., 'none' or 'debye'). If set, only CSVs with this model are used.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional tag for the figure title.",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()

    indir = Path(args.indir).expanduser().resolve()
    if not indir.is_dir():
        raise NotADirectoryError(f"Not a directory: {indir}")

    theta_i_deg, summaries_by_case, meta = load_summaries_theta(
        indir=indir,
        freq_filter=args.freq_GHz,
        ferri_filter=args.ferri_model,
    )

    freq_list = meta["freq_GHz"]
    ferri_list = meta["ferri_model"]
    materials = meta["materials"]
    cases = meta["cases"]

    # Build a concise title
    if args.freq_GHz is not None:
        freq_str = f"{args.freq_GHz:g}"
    else:
        freq_str = ",".join(f"{f:g}" for f in freq_list) if freq_list else "?"

    if args.ferri_model is not None:
        ferri_str = args.ferri_model
    else:
        ferri_str = ",".join(sorted(set(ferri_list))) if ferri_list else "?"

    th_min = float(theta_i_deg[0])
    th_max = float(theta_i_deg[-1])

    tag = f" | {args.tag}" if args.tag else ""
    title = (
        f"Variable incidence angle: f={freq_str} GHz | θi={th_min:g}–{th_max:g}° | "
        f"ferri={ferri_str} | materials={len(materials)}{tag}"
    )

    out_png = Path(args.outpng).expanduser().resolve()
    plot_4x2_allmaterials_theta(theta_i_deg, summaries_by_case, out_png, title)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
