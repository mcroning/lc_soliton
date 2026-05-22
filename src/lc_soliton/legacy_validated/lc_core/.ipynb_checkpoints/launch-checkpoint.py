# lc_core/launch.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import numpy as np


@dataclass
class LaunchArrays:
    """
    GPU launch arrays produced by build_launch_field_gpu.
    """
    amp0: Any
    arrin: Any
    I0: Any
    norm0: Any
    support: Any


def _resolve_image_file(cfg):
    """
    Optional legacy image-mask resolver.

    Image application is intentionally retained but disabled by default.
    It is not part of the standard soliton path.
    """
    if cfg.launch.image_on_beam == "No Image":
        return None

    external_image = cfg.launch.external_image.strip()
    std_dir = Path(cfg.launch.std_image_dir).expanduser()
    std_map = {
        "MNIST 0": "mnist0.png", "MNIST 1": "mnist1.png", "MNIST 2": "mnist2.png",
        "MNIST 3": "mnist3.png", "MNIST 4": "mnist4.png", "MNIST 5": "mnist5.png",
        "MNIST 6": "mnist6.png", "MNIST 7": "mnist7.png", "MNIST 8": "mnist8.png",
        "MNIST 9": "mnist9.png", "AF Res Chart": "AF Res Chart.png",
    }

    image_file = Path(external_image).expanduser() if external_image else std_dir / std_map.get(cfg.launch.std_image, "")

    if (image_file is None) or (not image_file.exists()):
        raise FileNotFoundError(f"Image file not found: {image_file}")

    return image_file


def build_arrin_gpu(cfg, xp):
    """
    Build the optional image mask on GPU.

    xp is usually cupy. This is passed explicitly to avoid importing CuPy at
    module import time.
    """
    arrin = xp.ones((28, 28), dtype=xp.float32)

    image_file = _resolve_image_file(cfg)
    if image_file is None:
        return arrin

    try:
        from PIL import Image, ImageChops
        from scipy.ndimage import zoom as sp_zoom
    except Exception as e:
        raise RuntimeError("Image-mask support requires PIL and scipy.") from e

    img = Image.open(str(image_file)).convert("L")
    if cfg.launch.image_invert:
        img = ImageChops.invert(img)

    img_cpu = np.array(img, dtype=np.float32)

    imsizex, imsizey = img_cpu.shape
    maxsize = int(max(imsizex, imsizey))
    if maxsize % 2:
        maxsize += 1

    fill = float(img_cpu.max() if img_cpu.size else 1.0)
    image_sq = np.ones((maxsize, maxsize), dtype=np.float32) * fill

    ox = (maxsize - imsizex) // 2
    oy = (maxsize - imsizey) // 2
    image_sq[ox:ox + imsizex, oy:oy + imsizey] = img_cpu

    g = cfg.grid
    zoomsc = float(cfg.launch.image_size_factor) * (float(g.xaper_um) / 2.0) / maxsize
    zx = zoomsc * int(g.Nx) / float(g.xaper_um)
    zy = zoomsc * int(g.Ny) / float(g.yaper_um)

    arrin_cpu = sp_zoom(np.rot90(image_sq, k=1), (zx, zy), order=0).astype(np.float32, copy=False)

    m = float(arrin_cpu.max()) if arrin_cpu.size else 1.0
    if m > 0:
        arrin_cpu /= m

    return xp.asarray(arrin_cpu, dtype=xp.float32)


def build_launch_field_gpu(
    cfg,
    ctx,
    *,
    genrot,
    build_amp_pair,
    intens,
    fft_module=None,
):
    """
    Build the legacy two-component launch field on GPU.

    Required old validated functions are injected:
        genrot(...)
        build_amp_pair(...)
        intens(amp0, coh)

    This keeps this module clean while allowing direct reuse of old tested code.
    """
    cp = ctx.xp
    spfft = fft_module if fft_module is not None else cp.fft

    g = cfg.grid
    l = cfg.launch

    arrin = build_arrin_gpu(cfg, cp)

    ang_rad = math.radians(float(l.soliton_pair_angle_deg))

    xoffset1 = float(l.xoffset_um) + float(l.soliton_pair_sep_um) * math.cos(ang_rad)
    xoffset2 = float(l.xoffset_um) - float(l.soliton_pair_sep_um) * math.cos(ang_rad)
    yoffset1 = float(l.yoffset_um) + float(l.soliton_pair_sep_um) * math.sin(ang_rad)
    yoffset2 = float(l.yoffset_um) - float(l.soliton_pair_sep_um) * math.sin(ang_rad)

    z_focus = 0.0 if bool(l.soliton) else (float(g.rlen_um) / 2.0)

    coord1 = genrot(
        float(g.rlen_um),
        float(l.thout1),
        float(l.phi1),
        ctx.x + xoffset1,
        ctx.y + yoffset1,
        refin=float(ctx.refin),
        z_focus=z_focus,
    )

    coord2 = genrot(
        float(g.rlen_um),
        float(l.thout2),
        float(l.phi2),
        ctx.x + xoffset2,
        ctx.y + yoffset2,
        refin=float(ctx.refin),
        z_focus=z_focus,
    )

    # build_amp_pair still expects legacy prdata-like keys in many old versions.
    prdata_like = dict(cfg.legacy_prdata)
    prdata_like.update(
        dict(
            lm=float(g.lm_um),
            xsamp=int(g.Nx),
            ysamp=int(g.Ny),
            xaper=float(g.xaper_um),
            yaper=float(g.yaper_um),
            rlen=float(g.rlen_um),
            dz=float(g.dz_um),
            P=float(cfg.material.P_mW),
            coh=bool(l.coh),
            refin=float(ctx.refin),
            realimage=str(cfg.legacy_prdata.get("realimage", "amplitude image")),
            image_on_beam=str(cfg.launch.image_on_beam),
            w0x1=float(l.w0x1_um),
            w0y1=float(l.w0y1_um),
            w0x2=float(l.w0x2_um),
            w0y2=float(l.w0y2_um),
            w01x=float(l.w0x1_um),
            w01y=float(l.w0y1_um),            
            w02x=float(l.w0x2_um),            
            w02y=float(l.w0y2_um),
        )
    )

    
    amp0m, amp0p = build_amp_pair(arrin, coord1, coord2, prdata_like)
    amp0 = cp.asarray((amp0m, amp0p), dtype=cp.complex64)

    support = (float(g.lm_um) ** 2 * ctx.fxy2 < 1.0)

    A0 = spfft.fft2(amp0, axes=(-2, -1))
    A0 = cp.where(support[None, :, :], A0, 0)
    amp0 = spfft.ifft2(A0, axes=(-2, -1)).astype(cp.complex64, copy=False)

    I0 = intens(amp0, bool(l.coh)).astype(cp.float32, copy=False)
    norm0 = cp.sum(I0) * cp.float32(float(g.dx_um) * float(g.dy_um))
    norm0 = cp.where(norm0 == 0.0, 1.0, norm0)

    amp0 = (amp0 / cp.sqrt(norm0)).astype(cp.complex64, copy=False)
    I0 = intens(amp0, bool(l.coh)).astype(cp.float32, copy=False)

    return LaunchArrays(
        amp0=amp0,
        arrin=arrin,
        I0=I0,
        norm0=norm0,
        support=support,
    )
