#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Imports:
import numpy as np


def complex_sqrt(z):
    return np.sqrt(z + 0j)


def snell_cos_theta_t(n1, n2, cos_theta_i):
    # sin^2(theta_t) = (n1/n2)^2 * (1 - cos^2 theta_i)
    sin2_t = (n1 / n2)**2 * (1.0 - cos_theta_i**2)
    cos2_t = 1.0 - sin2_t
    return complex_sqrt(cos2_t)


def fresnel_coeffs_magnetic(eta1, eta2, n1, n2, cos_theta_i):
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
    return rs, rp


def circular_components_from_rs_rp(rs, rp):
    oc = 0.5 * (rs + rp)
    sc = 0.5 * (rs - rp)
    return np.abs(oc)**2, np.abs(sc)**2


def nearest(arr, target):
    arr = np.asarray(arr)
    return arr[np.argmin(np.abs(arr - target))]
