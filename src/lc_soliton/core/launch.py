from __future__ import annotations

__all__ = [
    "intensity",
    "make_input_field",
]


def make_input_field(p: LCParams, x, y, xp):
    X = x[:, None]
    Y = y[None, :]
    g1 = xp.exp(-(X**2) / float(p.waist_x_um) ** 2 - ((Y - 0.5 * p.y_sep_um) ** 2) / float(p.waist_y_um) ** 2)
    if abs(float(p.y_sep_um)) > 0:
        g2 = xp.exp(-(X**2) / float(p.waist_x_um) ** 2 - ((Y + 0.5 * p.y_sep_um) ** 2) / float(p.waist_y_um) ** 2)
    else:
        g2 = xp.zeros_like(g1)

    amp = xp.stack([g1, g2], axis=0).astype(xp.complex64)
    norm = xp.sum(intensity(amp, p.coherent, xp=xp)) * xp.float32((p.xaper_um / p.Nx) * (p.yaper_um / p.Ny))
    norm = xp.where(norm == 0, xp.float32(1.0), norm)
    return (amp / xp.sqrt(norm / xp.float32(p.power_norm))).astype(xp.complex64)

def intensity(amp, coherent: bool, *, xp):
    a0, a1 = amp[0], amp[1]
    if coherent:
        s = a0 + a1
        return (s.real * s.real + s.imag * s.imag).astype(xp.float32)
    return (a0.real * a0.real + a0.imag * a0.imag + a1.real * a1.real + a1.imag * a1.imag).astype(xp.float32)