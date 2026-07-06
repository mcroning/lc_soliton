"""LC/optical coupling coefficients."""

from __future__ import annotations

from .liquid_crystal import LCSpec
from scipy.constants import c as C0

def compute_bi_from_power(P_mW, *, d_um=75.0, K=7e-12, ne=1.7, no=1.5):
    d_m = float(d_um) * 1e-6
    P_W = float(P_mW) * 1e-3
    na2 = float(ne) ** 2 - float(no) ** 2
    return float(na2 * d_m**2 * 1e12 * P_W / (8.0 * C0 * float(K)))

def resolved_bi(lc, beams):
    P_mW = sum(float(ch.power) for ch in beams.channels)
    return compute_bi_from_power(
        P_mW,
        d_um=lc.cell.thickness_um,
        K=lc.material.K_N,
        ne=lc.material.ne,
        no=lc.material.no,
    )
