# Radar and Magnetism (RadMag) Python Package

## Description
These are useful tools for radar theory modeling and analysis. A full test script with tunable parameters is included.


## Installation

These tools can be installed as a Python package called "radmagpy" for easy access to functions and visualization tools. This was done so that one could organize files into separate directories and still be able to reference each file by defining a path relative to the package root name.

To install as a local branch (highly recommended. NOTE the "-e" flag to install as a local branch! Again, highly recommended you do not ignore the "-e" flag!)

```
pip install -e <path/to/your/local/python/package/directory containing setup.py>
```
NOTE: This is not a path to setup.py! This is the path to the root directory containing setup.py!

As an example, if you are user tycho and you want to use and edit your copy of the radmagpy package that you put in `/homes/tycho/code/radmagpy/src`, you would do:
```
pip install -e /homes/tycho/code/radmagpy/
```


## Usage
### No roughness model
To run with no roughness model, i.e. no geometric depolarization
```
python full_test.py --roughness-model 'none' --outdir 'results'
```

### Facet-only roughness model
To run with a facet-only model, using the RMS slopes only
```
python full_test.py --roughness-model 'facet' --outdir 'results'
```

### Simple roughness model
To run with a simple roughness-driven depolarization of `k ~ 0.5·slope²`
```
python full_test.py --roughness-model 'simple' --outdir 'results'
```

### Simplified Integral Equation Model (IEM)
To run with a frequency-dependent single-parameter depolarizer using
`F = exp[-(4πσ_h cosθ / λ)^2]`

A fraction `(1 - F)` of **OC** power is transferred into **SC**, plus an optional **dihedral** term that enhances SC:

- `OC_new = F * OC`
- `SC_new = SC + (1 - F) * OC + k_d * sin^2(θ) * OC_new`

where:

| Symbol | Meaning |
|--------|---------|
| `σ_h`  | RMS height of surface roughness (meters) |
| `θ`    | Local incidence angle for each facet |
| `λ`    | Radar wavelength |
| `F`    | Smooth-surface coherent return factor |
| `k_d`  | Optional dihedral boost coefficient |
```
python full_test.py --roughness-model 'iem-lite' --outdir 'results'
```

### IEM single-bounce (monostatic) term model

To run with a single-bounce form for co-pol backscatter, using a surface PSD `W(q)` and Fresnel composites:

`σ⁰_pp(θ) ≈ [ k₀² cos²θ / 2 ] · |B_pp(θ)|² · W(2k₀ sinθ) · exp( -4 k₀² σ_h² cos²θ ) · S(θ)`

where:

- `k₀ = 2π / λ`
- `q = 2k₀ sinθ`
- `W(q)` = surface power spectral density (e.g., Gaussian or Exponential from `σ_h`,`L`)
- `B_pp(θ)` = Fresnel composite for polarization p∈{hh,vv} with complex ε*, μ*
- `S(θ)` = optional shadowing factor (e.g., `S(θ) = exp( - tan²θ / m )`)
- `σ_h` = RMS height; L = correlation length

Example PSDs:
- Gaussian:    `W_G(q) = (σ_h² L² / 4π) · exp( - (qL/2)² )`
- Exponential: `W_E(q) = (2 σ_h² L²) / (1 + q² L²)^(3/2)`

Cross-pol can be approximated pragmatically as:
`σ⁰_hv(θ) ≈ α · [ k₀ σ_h sinθ ]² · √( σ⁰_hh · σ⁰_vv )`

Map linear to circular under random azimuth:
- `OC = 0.5 · (σ⁰_hh + σ⁰_vv + 2 σ⁰_hv)`
- `SC = 0.5 · (σ⁰_hh + σ⁰_vv - 2 σ⁰_hv)`
```
python full_test.py --roughness-model 'iem' --outdir 'results'
```

### Hagfors scattering law

To run with an angular law with tunable curvature and exponent:

`σ⁰_hag(θ) = ρ₀ · cosⁿ(θ) / [ 1 + tan²(θ) / C ]^((n+1)/2)`

Optional polarization mix to split into OC/SC:
`OC = (1 - β) · σ⁰_hag(θ)`
`SC = β · σ⁰_hag(θ)`

where:
- `ρ₀` = amplitude scale
- `C`   = curvature parameter
- `n`   = cosine exponent
- `β`   = polarization-mixing fraction (0…1)
```
python full_test.py --roughness-model 'hagfors' --outdir 'results'
```


## Visuals
TBD


## Authors and acknowledgment
Primary Author: [Dany Waller](danywaller.github.io)
- Email: [dany.c.waller@gmail.com](mailto:dany.c.waller@gmail.com)
