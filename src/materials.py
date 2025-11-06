# ----------------------------- Materials -----------------------------

MATERIAL_CLASSES = {
    "Basaltic rock": {
        "eps_real": (6.0, 9.0),
        "tan_e": (5e-3, 5e-2),
        "sigma": (0.0, 1e-3),
        "mu_real": (0.98, 1.08),
        "tan_m": (1e-4, 5e-3),
        "ferrimag": False,
    },
    "Anorthositic rock": {
        "eps_real": (3.0, 5.0),
        "tan_e": (2e-3, 2e-2),
        "sigma": (0.0, 5e-4),
        "mu_real": (0.98, 1.05),
        "tan_m": (1e-4, 2e-3),
        "ferrimag": False,
    },
    "Porous regolith/soil": {
        "eps_real": (1.6, 3.5),
        "tan_e": (1e-3, 1e-2),
        "sigma": (0.0, 2e-4),
        "mu_real": (0.98, 1.10),
        "tan_m": (5e-5, 2e-3),
        "ferrimag": False,
    },
    "Water ice (clean/cold)": {
        "eps_real": (3.05, 3.25),
        "tan_e": (3e-4, 2e-3),
        "sigma": (0.0, 1e-6),
        "mu_real": (0.995, 1.01),
        "tan_m": (5e-5, 5e-4),
        "ferrimag": False,
    },
    "Salts/evaporites": {
        "eps_real": (4.5, 7.5),
        "tan_e": (1e-3, 5e-2),
        "sigma": (0.0, 2e-3),
        "mu_real": (0.98, 1.05),
        "tan_m": (1e-4, 2e-3),
        "ferrimag": False,
    },
    "Metal-rich regolith": {
        "eps_real": (5.0, 12.0),
        "tan_e": (5e-3, 5e-2),
        "sigma": (1e-3, 5e-2),
        "mu_real": (0.98, 1.20),
        "tan_m": (5e-4, 1e-2),
        "ferrimag": False,
    },
    "Ferrimagnetic soil": {
        "eps_real": (4.0, 9.0),
        "tan_e": (2e-3, 3e-2),
        "sigma": (0.0, 5e-3),
        "mu_real": (1.05, 1.50),
        "tan_m": (1e-3, 2e-2),
        "ferrimag": True, #  (Fe-oxide/npFe)
    },
}
