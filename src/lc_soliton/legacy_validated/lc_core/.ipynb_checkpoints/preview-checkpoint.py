from __future__ import annotations

import math
import numpy as np


def make_cpu_grid(cfg):
    g = cfg.grid
    x = (np.arange(g.Nx, dtype=np.float32) - g.Nx / 2) * g.dx_um + 0.5 * g.dx_um
    y = (np.arange(g.Ny, dtype=np.float32) - g.Ny / 2) * g.dy_um + 0.5 * g.dy_um
    return x, y


def pair_offsets(cfg):
    l = cfg.launch
    ang = math.radians(l.soliton_pair_angle_deg)

    x1 = l.xoffset_um + l.soliton_pair_sep_um * math.cos(ang)
    x2 = l.xoffset_um - l.soliton_pair_sep_um * math.cos(ang)
    y1 = l.yoffset_um + l.soliton_pair_sep_um * math.sin(ang)
    y2 = l.yoffset_um - l.soliton_pair_sep_um * math.sin(ang)

    return (x1, y1), (x2, y2)


def sponge_window_cpu(cfg):
    g = cfg.grid
    alpha = cfg.boundary.windowedge

    try:
        from scipy.signal.windows import tukey
        wx = tukey(g.Nx, alpha=alpha, sym=False).astype(np.float32)
        wy = tukey(g.Ny, alpha=alpha, sym=False).astype(np.float32)
    except Exception:
        wx = np.ones(g.Nx, dtype=np.float32)
        wy = np.ones(g.Ny, dtype=np.float32)

    if not cfg.boundary.use_sponge:
        return np.ones((g.Nx, g.Ny), dtype=np.float32)

    return np.sqrt(np.outer(wx, wy)).astype(np.float32)


def preview_launch_geometry(cfg, *, print_summary=True):
    x, y = make_cpu_grid(cfg)
    (x1, y1), (x2, y2) = pair_offsets(cfg)
    windowxy = sponge_window_cpu(cfg)

    info = dict(
        x_range_um=(float(x.min()), float(x.max())),
        y_range_um=(float(y.min()), float(y.max())),
        beam1_center_um=(float(x1), float(y1)),
        beam2_center_um=(float(x2), float(y2)),
        separation_center_to_center_um=float(
            math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        ),
        coherent=bool(cfg.launch.coh),
        sponge_minmax=(float(windowxy.min()), float(windowxy.max())),
    )

    if print_summary:
        print("Launch preview")
        print("--------------")
        for k, v in info.items():
            print(f"{k}: {v}")

    return info, x, y, windowxy

def gaussian_pair_preview(cfg):
    import numpy as np

    x, y = make_cpu_grid(cfg)
    X, Y = np.meshgrid(x, y, indexing="ij")

    (x1, y1), (x2, y2) = pair_offsets(cfg)

    w0 = 2.0

    I1 = np.exp(-((X - x1)**2 + (Y - y1)**2) / w0**2)
    I2 = np.exp(-((X - x2)**2 + (Y - y2)**2) / w0**2)

    if cfg.launch.coh:
        E = np.sqrt(I1) + np.sqrt(I2)
        I = np.abs(E)**2
    else:
        I = I1 + I2

    return x, y, I