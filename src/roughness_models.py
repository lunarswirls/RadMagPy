#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np

C0   = 299_792_458.0

# --- IEM spectral density W(q) for common correlation models ---
def psd_gaussian(q, sigma_h, L):
    # Gaussian correlation: C(r)=sigma^2 exp(-(r/L)^2) -> PSD W(q) = (sigma^2 L^2 / 4π) * exp(-(q L/2)^2)
    return (sigma_h**2 * L**2 / (4.0*np.pi)) * np.exp(-0.25 * (q*L)**2)


def psd_exponential(q, sigma_h, L):
    # Exponential correlation: C(r)=sigma^2 exp(-r/L) -> W(q) = (2 sigma^2 L^2) / (1 + q^2 L^2)^(3/2)
    return (2.0 * sigma_h**2 * L**2) / np.power(1.0 + (q*L)**2, 1.5)


def iem_shadowing(theta_rad, m=2.0):
    # Simple empirical shadowing -> S(θ) ~ exp(- (tanθ)^2 / m)
    tanth = np.tan(theta_rad)
    return np.exp(-(tanth**2)/max(m, 1e-6))


def fresnel_Bpp_vv(eta1, eta2, n1, n2, cos_theta_i):
    """
    Composite Fresnel factor Bpp (vv, hh) for magnetic media (complex ε*, μ*)

    :param eta1:
    :param eta2:
    :param n1:
    :param n2:
    :param cos_theta_i:
    :return:
    """
    ct = snell_cos_theta_t(n1, n2, cos_theta_i)
    rs = (eta2 * cos_theta_i - eta1 * ct) / (eta2 * cos_theta_i + eta1 * ct)
    rp = (eta1 * cos_theta_i - eta2 * ct) / (eta1 * cos_theta_i + eta2 * ct)
    # In IEM single-bounce, the co-pol weighting uses nominal Fresnel magnitude at the local angle.
    # vv uses rp (E field in plane of incidence), hh uses rs.
    return rp, rs


def iem_single_bounce_OC_SC(theta_rad, freq_Hz, eps2, mu2,
                            sigma_h, L, psd_kind="gaussian",
                            shadow=False, shadow_m=2.0, cal_kx=1.0):
    """
    Returns (OC, SC) power fractions from an IEM single-bounce term,
    computed in *linear* basis first (hh/vv), then mapped to *circular* OC/SC.

    This is a pragmatic implementation that:
      - builds σ0_hh and σ0_vv from a classic IEM-like spectral form
      - includes an exp(-4 k^2 σ_h^2 cos^2 θ) coherent-damping term
      - optionally applies a shadowing factor S(θ)
      - estimates σ0_hv cross-pol as a small slope-driven fraction of the co-pol geometric mean

    Inputs:
      eps2, mu2 : complex permittivity/permeability of half-space
      sigma_h   : RMS height (meters)
      L         : correlation length (meters)
    """

    k0 = 2*np.pi*freq_Hz / C0
    q = 2*k0*np.sin(theta_rad)

    # PSD
    if psd_kind == "gaussian":
        Wq = psd_gaussian(q, sigma_h, L)
    elif psd_kind == "exponential":
        Wq = psd_exponential(q, sigma_h, L)
    else:
        raise ValueError("psd_kind must be 'gaussian' or 'exponential'")

    # Medium 1 (vacuum)
    eps1 = 1.0 + 0j
    mu1  = 1.0 + 0j
    eta1 = np.sqrt(mu1/eps1)
    n1   = np.sqrt(mu1*eps1)

    eta2 = np.sqrt(mu2/eps2)
    n2   = np.sqrt(mu2*eps2)
    ci   = np.cos(theta_rad)

    rp, rs = fresnel_Bpp_vv(eta1, eta2, n1, n2, ci)  # rp~vv, rs~hh

    # Core spectral factor
    spec = cal_kx * (k0**2) * (ci**2) * Wq
    coh  = np.exp(-4.0 * (k0**2) * (sigma_h**2) * (ci**2))  # coherent damping

    Sfac = iem_shadowing(theta_rad, m=shadow_m) if shadow else 1.0

    # Co-pol backscatter terms (monostatic)
    sigma_vv = 0.5 * spec * np.abs(rp)**2 * coh * Sfac
    sigma_hh = 0.5 * spec * np.abs(rs)**2 * coh * Sfac

    # Cross-pol (empirical, slope-driven fraction of geometric mean)
    # This captures depolarization growth with roughness without going to full SSA2 terms.
    slope2 = (k0 * sigma_h * np.sin(theta_rad))**2
    sigma_hv = 0.15 * slope2 * np.sqrt(sigma_vv * sigma_hh)

    # Map linear (hh, vv, hv) -> circular (OC, SC)
    # For monostatic, random azimuth assumption: OC ~ (σ_hh + σ_vv + 2 Re σ_hv_corr)/2
    # SC ~ (σ_hh + σ_vv - 2 Re σ_hv_corr)/2
    # We approximate <Re σ_hv_corr> ~ 0, and allocate cross-pol symmetrically.
    total_lin = sigma_hh + sigma_vv
    OC = 0.5 * (total_lin + 2*sigma_hv)
    SC = 0.5 * (total_lin - 2*sigma_hv)
    # Ensure non-negative
    OC = np.maximum(OC, 0.0)
    SC = np.maximum(SC, 0.0)
    return OC, SC


def hagfors_OC_SC(theta_rad, rho0=0.2, C=0.3, n=3.0, pol_mix=0.08):
    """
    Hagfors angular law for total backscatter power, split into OC/SC by a small mixing factor.
    σ0_hag(θ) = ρ0 * cos^n(θ) / (1 + tan^2(θ)/C)^((n+1)/2)

    pol_mix: fraction of total power routed to SC (rest to OC).
             Set ~0.0 for nearly coherent returns; increase to emulate micro-depolarization.
    """
    ct = np.cos(theta_rad)
    tt = np.tan(theta_rad)
    denom = np.power(1.0 + (tt**2)/max(C, 1e-12), 0.5*(n+1.0))
    sigma0 = rho0 * (ct**n) / denom
    sigma0 = np.maximum(sigma0, 0.0)

    SC = pol_mix * sigma0
    OC = (1.0 - pol_mix) * sigma0
    return OC, SC


def roughness_apply(OC, SC, model, *, theta_rad, rms_slope,
                    freq_Hz, sigma_h_m, dihedral_k,
                    rng, use_random_sigma_h=False, sigma_h_range=(0.003, 0.03),
                    eps2 = None, mu2 = None, corr_L = 0.03,
                    iem_psd = "gaussian", iem_shadow = False, iem_shadow_m = 2.0, iem_cal_kx = 1.0,
                    hag_C = 0.3, hag_n = 3.0, hag_rho0 = 0.2, hag_pol_mix = 0.08):
    """
    Returns (OCr, SCr) given chosen roughness model.

    Models:
      - 'none'   : return OC, SC as-is
      - 'facet'  : no extra OC->SC transfer; facet slopes already sampled
      - 'simple'   : k = 0.5 * rms_slope^2; transfer k*OC to SC
      - 'iem-lite'   : F = exp(-(4π σh cosθ / λ)^2); transfer (1-F)*OC to SC + dihedral term
      - 'iem'   : full IEM single-bounce term using (sigma_h, L)
      - 'hagfors'   : Hagfors angular law with OC/SC split via pol_mix
    """
    if model == "none" or model == "facet":
        return OC, SC

    if model == "simple":
        k = 0.5 * np.square(rms_slope)
        k = np.minimum(k, 0.8)
        OCn = (1 - k) * OC
        SCn = SC + k * OC
        return OCn, SCn

    if model == "iem-lite":
        lam = C0 / freq_Hz
        if use_random_sigma_h:
            sigma_h = rng.uniform(sigma_h_range[0], sigma_h_range[1], size=OC.shape)
        else:
            sigma_h = sigma_h_m

        # Smooth-surface power reduction (scalar), classic exp(-(4πσh cosθ / λ)^2)
        red = np.exp(-np.square(4*np.pi*sigma_h*np.cos(theta_rad)/lam))
        # Transfer a fraction (1 - red) of OC into SC (single-bounce depolarization)
        trans = np.clip(1.0 - red, 0.0, 0.95)
        OC1 = (1 - trans) * OC
        SC1 = SC + trans * OC

        # Add a simple dihedral boost scaling with sin^2(theta)
        SC2 = SC1 + dihedral_k * np.square(np.sin(theta_rad)) * OC1
        OC2 = OC1 * (1 - 0.25*dihedral_k)  # tiny energy bookkeeping
        return OC2, SC2

    # --- Full IEM single-bounce ---
    if model == "iem":
        lam = C0 / freq_Hz
        sigma_h = (rng.uniform(sigma_h_range[0], sigma_h_range[1], size=OC.shape)
                   if use_random_sigma_h else sigma_h_m)
        # correlation length; keep user-editable; you may also randomize
        L = corr_L
        OC_iem, SC_iem = iem_single_bounce_OC_SC(
            theta_rad, freq_Hz, eps2=eps2, mu2=mu2,
            sigma_h=sigma_h, L=L,
            psd_kind=iem_psd, shadow=iem_shadow, shadow_m=iem_shadow_m, cal_kx=iem_cal_kx
        )
        return OC_iem, SC_iem

    # --- Hagfors ---
    if model == "hagfors":
        OC_h, SC_h = hagfors_OC_SC(theta_rad, rho0=hag_rho0, C=hag_C, n=hag_n, pol_mix=hag_pol_mix)
        return OC_h, SC_h

    raise ValueError(f"Unknown roughness model: {model}")