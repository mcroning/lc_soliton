from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .backend import asnumpy
from .context import LCContext

# -----------------------------------------------------------------------------
# Lightweight storage
# -----------------------------------------------------------------------------


class LightStore:
    def __init__(self, run_dir: Path, ctx: LCContext, *, Nt_out: int, save_slices: bool = True, save_full: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = ctx
        self.Nt_out = int(Nt_out)
        self.save_slices = bool(save_slices)
        self.save_full = bool(save_full)
        self.log_path = self.run_dir / "scalar_log.csv"
        self.log_file = open(self.log_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.log_file,
            fieldnames=["jt", "k", "z_um", "Imax", "power", "theta_min", "theta_max", "dtheta_max", "rrms", "rmax", "converged"],
        )
        self.writer.writeheader()

        if self.save_slices:
            self.Ixz = np.memmap(self.run_dir / "Ixz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Nx))
            self.Iyz = np.memmap(self.run_dir / "Iyz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Ny))
            self.dthetaxz = np.memmap(self.run_dir / "dthetaxz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Nx))
            self.dthetayz = np.memmap(self.run_dir / "dthetayz.dat", dtype=np.float32, mode="w+", shape=(Nt_out, ctx.Nz, ctx.Ny))
        if self.save_full:
            self.theta_final = np.memmap(self.run_dir / "theta_final.dat", dtype=np.float32, mode="w+", shape=(ctx.Nz, ctx.Nx, ctx.Ny))

    def save(self, slot: int, k: int, I, theta, info: Dict[str, Any]) -> None:
        ctx = self.ctx
        Icpu = asnumpy(I).astype(np.float32, copy=False)
        thcpu = asnumpy(theta).astype(np.float32, copy=False)
        bcpu = asnumpy(ctx.theta_bias_2d).astype(np.float32, copy=False)

        if self.save_slices:
            self.Ixz[slot, k, :] = Icpu[:, ctx.Ny // 2]
            self.Iyz[slot, k, :] = Icpu[ctx.Nx // 2, :]
            dth = thcpu - bcpu
            self.dthetaxz[slot, k, :] = dth[:, ctx.Ny // 2]
            self.dthetayz[slot, k, :] = dth[ctx.Nx // 2, :]
        if self.save_full:
            self.theta_final[k] = thcpu

        self.writer.writerow(
            dict(
                jt=slot,
                k=k,
                z_um=k * ctx.dz,
                Imax=float(Icpu.max()),
                power=float(Icpu.sum() * ctx.dx * ctx.dy),
                theta_min=float(thcpu.min()),
                theta_max=float(thcpu.max()),
                dtheta_max=float(np.max(np.abs(thcpu - bcpu))),
                rrms=float(info.get("rrms", np.nan)),
                rmax=float(info.get("rmax", np.nan)),
                converged=bool(info.get("converged", False)),
            )
        )

    def close(self) -> None:
        self.log_file.flush()
        self.log_file.close()
        if self.save_slices:
            self.Ixz.flush(); self.Iyz.flush(); self.dthetaxz.flush(); self.dthetayz.flush()
        if self.save_full:
            self.theta_final.flush()


__all__ = [
    "LightStore",
]