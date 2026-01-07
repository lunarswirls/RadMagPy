import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _bootstrap_ci_median(
    x: np.ndarray,
    ci: float = 0.95,
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Returns (median, lo, hi) where lo/hi are bootstrap CI bounds for the median.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(0)

    med = float(np.nanmedian(x))
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    meds = np.nanmedian(x[idx], axis=1)

    alpha = 1.0 - ci
    lo = float(np.nanquantile(meds, alpha / 2.0))
    hi = float(np.nanquantile(meds, 1.0 - alpha / 2.0))
    return med, lo, hi


def _percentile_band(x: np.ndarray, lo_q: float, hi_q: float) -> tuple[float, float, float]:
    """
    Returns (median, lo, hi) where lo/hi are distribution quantiles (NOT a CI of the median).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    med = float(np.nanmedian(x))
    lo = float(np.nanquantile(x, lo_q))
    hi = float(np.nanquantile(x, hi_q))
    return med, lo, hi


def compute_group_median_with_ci(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    *,
    ci: float = 0.95,
    method: str = "bootstrap",  # "bootstrap" or "percentile"
    n_boot: int = 1000,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """
    Compute median and uncertainty band per group.
    Output columns: group_cols + ["med", "lo", "hi", "n"].
    """
    rng = np.random.default_rng(rng_seed)

    rows: list[dict] = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        x = sub[value_col].to_numpy()

        if method == "bootstrap":
            med, lo, hi = _bootstrap_ci_median(x, ci=ci, n_boot=n_boot, rng=rng)
        elif method == "percentile":
            alpha = 1.0 - ci
            med, lo, hi = _percentile_band(x, lo_q=alpha / 2.0, hi_q=1.0 - alpha / 2.0)
        else:
            raise ValueError("method must be 'bootstrap' or 'percentile'")

        row = dict(zip(group_cols, keys))
        row.update({"med": med, "lo": lo, "hi": hi, "n": int(np.isfinite(x).sum())})
        rows.append(row)

    return pd.DataFrame(rows)


def make_freq_summary_figure(
    df: pd.DataFrame,
    *,
    outdir: str,
    roughness_model: str,
    cpr_cap: float | None = None,
    class_col: str = "class",
    freq_col: str = "freq_Hz",
    cpr_col: str = "CPR_stable",
    oc_col: str = "OC",
    sc_col: str = "SC",
    ci: float = 0.95,
    ci_method: str = "bootstrap",   # "bootstrap" (CI on median) or "percentile" (data band)
    n_boot: int = 800,
    rng_seed: int = 0,
    show_ci: bool = True,
    legend_ncol: int = 4,
    figsize: tuple[float, float] = (12, 6),
    dpi: int = 180,
    filename_prefix: str = "CPR_OC_SC_vs_freq",
) -> str:
    """
    Save a 3-panel figure (1x3) of CPR/OC/SC vs frequency by class, using
    median lines and optional CI/band shading.

    Returns the saved PNG path.
    """
    os.makedirs(outdir, exist_ok=True)

    group_cols = [class_col, freq_col]

    cpr_stats = compute_group_median_with_ci(
        df, cpr_col, group_cols, ci=ci, method=ci_method, n_boot=n_boot, rng_seed=rng_seed
    ).rename(columns={"med": "CPR_med", "lo": "CPR_lo", "hi": "CPR_hi", "n": "CPR_n"})

    oc_stats = compute_group_median_with_ci(
        df, oc_col, group_cols, ci=ci, method=ci_method, n_boot=n_boot, rng_seed=rng_seed
    ).rename(columns={"med": "OC_med", "lo": "OC_lo", "hi": "OC_hi", "n": "OC_n"})

    sc_stats = compute_group_median_with_ci(
        df, sc_col, group_cols, ci=ci, method=ci_method, n_boot=n_boot, rng_seed=rng_seed
    ).rename(columns={"med": "SC_med", "lo": "SC_lo", "hi": "SC_hi", "n": "SC_n"})

    # Merge into one table for convenience
    g = cpr_stats.merge(oc_stats, on=group_cols, how="outer").merge(sc_stats, on=group_cols, how="outer")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    classes = g[class_col].dropna().unique()

    for cls in classes:
        d = g[g[class_col] == cls].sort_values(freq_col)
        x = d[freq_col].to_numpy() / 1e9

        # CPR
        axes[0].plot(x, d["CPR_med"].to_numpy(), marker="o", label=str(cls))
        if show_ci:
            axes[0].fill_between(x, d["CPR_lo"].to_numpy(), d["CPR_hi"].to_numpy(), alpha=0.18)

        # OC
        axes[1].plot(x, d["OC_med"].to_numpy(), marker="o")
        if show_ci:
            axes[1].fill_between(x, d["OC_lo"].to_numpy(), d["OC_hi"].to_numpy(), alpha=0.18)

        # SC
        axes[2].plot(x, d["SC_med"].to_numpy(), marker="o")
        if show_ci:
            axes[2].fill_between(x, d["SC_lo"].to_numpy(), d["SC_hi"].to_numpy(), alpha=0.18)

    # Labels/titles
    cpr_lab = "Median CPR"
    if cpr_cap is not None and cpr_cap > 0:
        cpr_lab += f" (capped at {cpr_cap:g})"

    band_label = f"{int(ci*100)}% " + ("CI (median)" if ci_method == "bootstrap" else "band (quantiles)")
    if not show_ci:
        band_label = ""

    axes[0].set_ylabel(cpr_lab)
    axes[0].set_xlabel("Frequency (GHz)")
    axes[0].set_title(f"Median CPR vs Frequency\n{roughness_model}" + (f"\n{band_label}" if show_ci else ""))

    axes[1].set_ylabel("Median OC")
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_title(f"Median OC vs Frequency\n{roughness_model}" + (f"\n{band_label}" if show_ci else ""))

    axes[2].set_ylabel("Median SC")
    axes[2].set_xlabel("Frequency (GHz)")
    axes[2].set_title(f"Median SC vs Frequency\n{roughness_model}" + (f"\n{band_label}" if show_ci else ""))

    for ax in axes:
        ax.grid(True, alpha=0.3)

    # Shared legend below figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize="small", ncol=legend_ncol,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    safe_model = str(roughness_model).lower().replace(" ", "_")
    out_name = f"{filename_prefix}_{safe_model}_ci-{ci_method}.png"
    out_path = os.path.join(outdir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def make_cpr_heatmaps(
    df,
    *,
    outdir: str,
    roughness_model: str,
    target_freqs_GHz: tuple[float, float, float] = (0.85, 2.37, 7.14),
    freq_col: str = "freq_Hz",
    mu_col: str = "mu2_real",
    eps_col: str = "eps2_real",
    cpr_col: str = "CPR_stable",
    nbins: int = 28,
    cpr_cap: float | None = None,
    figsize: tuple[float, float] = (14, 5),
    dpi: int = 180,
    filename_prefix: str = "CPR_heatmaps_mu_eps",
) -> str:
    """
    Make a 1x3 panel of median CPR heatmaps across (mu', eps') at three target frequencies,
    using the nearest available sampled frequencies in df[freq_col].

    Color scale is consistent across all subplots:
      vmin = 0
      vmax = cpr_cap (if provided) else max over all 3 subplots
    """
    os.makedirs(outdir, exist_ok=True)

    def _nearest(arr, target):
        arr = np.asarray(arr)
        return arr[np.argmin(np.abs(arr - target))]

    def _median_cpr_grid(dfin, f_Hz, nbins=28):
        sub = dfin[np.isclose(dfin[freq_col], f_Hz)]
        if sub.empty:
            return None, None, None

        mu = sub[mu_col].to_numpy()
        epsr = sub[eps_col].to_numpy()
        cpr = sub[cpr_col].to_numpy()

        good = np.isfinite(mu) & np.isfinite(epsr) & np.isfinite(cpr)
        mu, epsr, cpr = mu[good], epsr[good], cpr[good]
        if mu.size == 0:
            return None, None, None

        mu_edges = np.linspace(np.nanmin(mu), np.nanmax(mu), nbins + 1)
        eps_edges = np.linspace(np.nanmin(epsr), np.nanmax(epsr), nbins + 1)

        grid = np.full((nbins, nbins), np.nan)
        for i in range(nbins):
            for j in range(nbins):
                m = (
                    (mu >= mu_edges[i]) & (mu < mu_edges[i + 1]) &
                    (epsr >= eps_edges[j]) & (epsr < eps_edges[j + 1])
                )
                if np.any(m):
                    grid[i, j] = np.nanmedian(cpr[m])

        mu_cent = 0.5 * (mu_edges[:-1] + mu_edges[1:])
        eps_cent = 0.5 * (eps_edges[:-1] + eps_edges[1:])
        return mu_cent, eps_cent, grid

    unique_freqs = np.sort(df[freq_col].unique())

    # Resolve target freqs to nearest sampled freqs
    chosen_freqs = [_nearest(unique_freqs, fGHz * 1e9) for fGHz in target_freqs_GHz]

    # Build grids and compute shared vmax
    grids = []
    max_over_all = 0.0  # enforce vmin=0
    for f_sel in chosen_freqs:
        mu_cent, eps_cent, grid = _median_cpr_grid(df, f_sel, nbins=nbins)
        grids.append((f_sel, mu_cent, eps_cent, grid))
        if grid is not None:
            gmax = np.nanmax(grid)
            if np.isfinite(gmax):
                max_over_all = max(max_over_all, float(gmax))

    # Shared color scaling
    vmin = 0.0
    if cpr_cap is not None and cpr_cap > 0:
        vmax = float(cpr_cap)
    else:
        vmax = max_over_all if np.isfinite(max_over_all) and max_over_all > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    # Plot each panel
    im_last = None
    for ax, (f_sel, mu_cent, eps_cent, grid) in zip(axes, grids):
        if grid is None:
            ax.set_axis_off()
            ax.set_title(f"No data at ~{f_sel/1e9:.2f} GHz")
            continue

        grid_plot = grid.copy()

        # If cpr_cap is set, cap values for display (consistent with vmax)
        if cpr_cap is not None and cpr_cap > 0:
            grid_plot = np.where(grid_plot > cpr_cap, cpr_cap, grid_plot)

        extent = [eps_cent.min(), eps_cent.max(), mu_cent.min(), mu_cent.max()]
        im_last = ax.imshow(
            np.flipud(grid_plot),
            aspect="auto",
            extent=extent,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel("ε' (real permittivity)")
        ax.set_ylabel("μ' (real permeability)")
        ax.set_title(f"Median CPR across (μ', ε')\n~{f_sel/1e9:.2f} GHz\n{roughness_model}")

    # Single shared horizontal colorbar at bottom
    if im_last is not None:
        cbar = fig.colorbar(
            im_last,
            ax=axes,
            orientation="horizontal",
            fraction=0.08,  # height of colorbar
            pad=0.18  # space between plots and colorbar
        )
        lab = "Median CPR"
        if cpr_cap is not None and cpr_cap > 0:
            lab += f" (capped at {cpr_cap:g})"
        cbar.set_label(lab)

    # Leave room at bottom for colorbar
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    safe_model = str(roughness_model).lower().replace(" ", "_")
    out_name = (
        f"{filename_prefix}_{safe_model}_"
        f"{target_freqs_GHz[0]:.2f}-{target_freqs_GHz[1]:.2f}-{target_freqs_GHz[2]:.2f}GHz.png"
    )
    out_path = os.path.join(outdir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def make_cpr_vs_mu_scatter(
    df,
    *,
    outdir: str,
    roughness_model: str,
    target_freq_GHz: float = 2.38,
    freq_col: str = "freq_Hz",
    mu_col: str = "mu2_real",
    eps_col: str = "eps2_real",
    cpr_col: str = "CPR_stable",
    alpha: float = 0.25,
    base_marker_size: float = 4.0,
    size_scale: float = 3.0,
    figsize: tuple[float, float] = (6.5, 5.5),
    dpi: int = 180,
    filename_prefix: str = "CPR_vs_mu",
) -> str:
    """
    Scatter plot of CPR vs mu' at a target frequency (default: S-band ~2.38 GHz),
    with marker size scaled by eps'.

    Saves one PNG and returns its path.
    """
    os.makedirs(outdir, exist_ok=True)

    def _nearest(arr, target):
        arr = np.asarray(arr)
        return arr[np.argmin(np.abs(arr - target))]

    unique_freqs = np.sort(df[freq_col].unique())
    f_sel = _nearest(unique_freqs, target_freq_GHz * 1e9)

    sub = df[np.isclose(df[freq_col], f_sel)].copy()
    if sub.empty:
        raise ValueError(f"No data found near {target_freq_GHz:.2f} GHz")

    mu = sub[mu_col].to_numpy()
    epsr = sub[eps_col].to_numpy()
    cpr = sub[cpr_col].to_numpy()

    good = np.isfinite(mu) & np.isfinite(epsr) & np.isfinite(cpr)
    mu, epsr, cpr = mu[good], epsr[good], cpr[good]

    # Marker size scaled by eps'
    eps_min, eps_max = np.nanmin(epsr), np.nanmax(epsr)
    ms = base_marker_size + size_scale * (epsr - eps_min) / (eps_max - eps_min + 1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(mu, cpr, s=ms, alpha=alpha)

    ax.set_xlabel("μ' (real permeability)")
    ax.set_ylabel("CPR (stable)")
    ax.set_title(
        f"CPR vs μ' at ~{f_sel/1e9:.2f} GHz\n"
        f"{roughness_model} (marker size ~ ε')"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_model = str(roughness_model).lower().replace(" ", "_")
    out_name = f"{filename_prefix}_{safe_model}_{f_sel/1e9:.2f}GHz.png"
    out_path = os.path.join(outdir, out_name)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path

