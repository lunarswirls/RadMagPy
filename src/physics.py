#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np

EPS0 = 8.854187817e-12
MU0  = 4*np.pi*1e-7
C0   = 299_792_458.0


def eps_complex_dispersion(eps_real0, tan_e0, sigma, f, f0=1.0e9):
    """
    ε*(f) = ε' - i( tanδe(f)*ε' + σ/(ωε0) ), with tanδe ~ baseline * (1 + a*(f/f0)^b)
    """
    a, b = 0.5, 0.3
    tan_e_f = tan_e0 * (1.0 + a * (f/f0)**b)
    eps_im_sigma = sigma / (2*np.pi*f*EPS0)
    return eps_real0 - 1j*(tan_e_f*eps_real0 + eps_im_sigma)


def mu_complex_dispersion(mu_real0, tan_m0, f, f0=1.0e9, ferri=False, rng=None):
    """
    μ*(f) = μ' - i tanδm(f) μ', with tanδm ~ baseline * (1 + c*(f/f0)^d)
    Optionally add a weak Lorentzian magnetic loss bump for ferrimagnetic soils.
    """
    c, d = 0.4, 0.2
    tan_m_f = tan_m0 * (1.0 + c * (f/f0)**d)
    mu_c = mu_real0 - 1j*tan_m_f*mu_real0
    if ferri:
        if rng is None:
            rng = np.random.default_rng(0)
        fr = 1.5e9 + 1.5e9 * rng.random(mu_real0.shape)  # 1.5–3.0 GHz
        gamma = 0.6e9
        # Lorentz term (very mild) – applied to μ'' only
        lorentz = (gamma**2) / ((2*np.pi*(f - fr))**2 + gamma**2)
        mu_c = (mu_c.real) - 1j*(mu_c.real * (tan_m_f + 0.02*lorentz))
    return mu_c
