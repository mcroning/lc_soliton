from __future__ import annotations

import math

__all__ = [
    "intensity",
    "make_input_field",
]


def _gaussian_beam(X, Y, *, x0, y0, waist_x, waist_y, xp):
    return xp.exp(
        -((X - xp.float32(x0)) ** 2) / xp.float32(float(waist_x) ** 2)
        -((Y - xp.float32(y0)) ** 2) / xp.float32(float(waist_y) ** 2)
    )


def _tilt_phase(X, Y, *, theta_deg, phi_deg, kout, refin, xp):
    theta = math.radians(float(theta_deg))
    phi = math.radians(float(phi_deg))

    k = float(kout) * float(refin)
    kx = k * math.sin(theta) * math.cos(phi)
    ky = k * math.sin(theta) * math.sin(phi)

    return xp.exp(
        1j * (
            xp.float32(kx) * X
            + xp.float32(ky) * Y
        )
    ).astype(xp.complex64)


def make_input_field(p, x, y, xp):
    X = x[:, None]
    Y = y[None, :]

    sep = float(getattr(p, "y_sep_um", 0.0))
    pair_angle = math.radians(float(getattr(p, "pair_angle_deg", 0.0)))

    # pair_angle = 0 means separation along y, matching the old y_sep_um behavior.
    dx_sep = 0.5 * sep * math.sin(pair_angle)
    dy_sep = 0.5 * sep * math.cos(pair_angle)

    waist_x1 = float(getattr(p, "waist_x_um", 8.0))
    waist_y1 = float(getattr(p, "waist_y_um", 8.0))
    waist_x2 = float(getattr(p, "waist_x2_um", waist_x1))
    waist_y2 = float(getattr(p, "waist_y2_um", waist_y1))

    power_ratio = max(0.0, float(getattr(p, "power_ratio", 0.0)))

    G1 = _gaussian_beam(
        X,
        Y,
        x0=dx_sep,
        y0=dy_sep,
        waist_x=waist_x1,
        waist_y=waist_y1,
        xp=xp,
    )

    G2 = _gaussian_beam(
        X,
        Y,
        x0=-dx_sep,
        y0=-dy_sep,
        waist_x=waist_x2,
        waist_y=waist_y2,
        xp=xp,
    )

    P1_frac = 1.0 / (1.0 + power_ratio)
    P2_frac = power_ratio / (1.0 + power_ratio)

    A1 = xp.sqrt(xp.float32(P1_frac)) * G1 * _tilt_phase(
        X,
        Y,
        theta_deg=getattr(p, "theta_out1_deg", 0.0),
        phi_deg=getattr(p, "phi1_deg", 0.0),
        kout=p.kout if hasattr(p, "kout") else 2.0 * math.pi / float(p.wavelength_um),
        refin=getattr(p, "refin", 1.0),
        xp=xp,
    )

    A2 = xp.sqrt(xp.float32(P2_frac)) * G2 * _tilt_phase(
        X,
        Y,
        theta_deg=getattr(p, "theta_out2_deg", 0.0),
        phi_deg=getattr(p, "phi2_deg", 0.0),
        kout=p.kout if hasattr(p, "kout") else 2.0 * math.pi / float(p.wavelength_um),
        refin=getattr(p, "refin", 1.0),
        xp=xp,
    )

    if power_ratio == 0.0:
        A2 = xp.zeros_like(A1)

    amp = xp.stack([A1, A2], axis=0).astype(xp.complex64)

    norm = xp.sum(intensity(amp, p.coherent, xp=xp)) * xp.float32(
        (p.xaper_um / p.Nx) * (p.yaper_um / p.Ny)
    )
    norm = xp.where(norm == 0, xp.float32(1.0), norm)

    return (amp / xp.sqrt(norm / xp.float32(p.power_norm))).astype(xp.complex64)


def intensity(amp, coherent: bool, *, xp):
    a0, a1 = amp[0], amp[1]

    if coherent:
        s = a0 + a1
        return (s.real * s.real + s.imag * s.imag).astype(xp.float32)

    return (
        a0.real * a0.real
        + a0.imag * a0.imag
        + a1.real * a1.real
        + a1.imag * a1.imag
    ).astype(xp.float32)