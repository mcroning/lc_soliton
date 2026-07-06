"""Clean soliton workflow.

A soliton here means a stationary self-consistent optical/director mode at a
specified total power.  This first clean implementation uses only the new
``lc`` package algorithms:

* backend/grid/launch/bias preparation,
* CN/Picard theta relaxation,
* split-step optical propagation,
* overlap-based beta estimate,
* field normalization and phase alignment.

This is intentionally compact and testable.  It is not a verbatim port of the
legacy polished eigensolver; it is the clean v001 solver to validate against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import time as _time

from ..request import StaticRequest
from ..result import BaseResult
from ..numerics.backend import get_backend, asnumpy, synchronize
from ..numerics.grid import make_grid
from ..physics.bias import build_bias
from ..physics.launch import build_launch
from ..physics.liquid_crystal import resolved_b, neff_from_theta
from ..physics.coupling import resolved_bi
from ..algorithms.splitstep import advance_slice, linear_kernel, total_intensity
from ..algorithms.theta_cn import prepare_cn_operator
from ..algorithms.theta_picard import cn_trapezoid_picard_step
from ..algorithms.thomas import solve_const_offdiag_batched
from ..products.diagnostics import intensity_metrics, residual_theta_static


@dataclass(frozen=True)
class SolitonRequest:
    """Request for one clean soliton solve."""

    base: StaticRequest
    max_outer: int = 100
    theta_steps_per_outer: int = 50
    field_mix: float = 0.5
    theta_mix: float = 1.0
    tol_field: float = 1e-6
    tol_theta: float = 1e-5
    tol_residual_rms: float = 5e-2
    tol_residual_max: float = 2e-1
    initial_A: Any | None = None
    initial_theta: Any | None = None


    def validate(self) -> None:
        self.base.validate()
        if self.max_outer < 1:
            raise ValueError("max_outer must be >= 1")
        if self.theta_steps_per_outer < 1:
            raise ValueError("theta_steps_per_outer must be >= 1")
        if not (0.0 < self.field_mix <= 1.0):
            raise ValueError("field_mix must be in (0, 1]")
        if not (0.0 < self.theta_mix <= 1.0):
            raise ValueError("theta_mix must be in (0, 1]")
        if self.tol_field <= 0.0:
            raise ValueError("tol_field must be positive")
        if self.tol_theta <= 0.0:
            raise ValueError("tol_theta must be positive")


@dataclass
class SolitonResult(BaseResult):
    """Result for one soliton solve."""

    A: Any | None = None
    theta: Any | None = None
    intensity: Any | None = None
    history: list[dict] = field(default_factory=list)
    converged: bool = False

def _prepare_initial_A(initial_A, launch, *, grid, target_power: float, coherent: bool, xp):
    if initial_A is None:
        A = launch.A0.copy()
    else:
        A = xp.asarray(initial_A, dtype=launch.A0.dtype).copy()
        if A.shape != launch.A0.shape:
            raise ValueError(f"initial_A shape {A.shape} does not match {launch.A0.shape}")
    return _normalize_power(A, target_power=target_power, grid=grid, coherent=coherent, xp=xp)


def _prepare_initial_theta(initial_theta, bias, *, xp):
    if initial_theta is None:
        theta = bias.theta_2d.copy()
    else:
        theta = xp.asarray(initial_theta, dtype=bias.theta_2d.dtype).copy()
        if theta.ndim == 3:
            theta = theta[int(theta.shape[0]) // 2].copy()
        if theta.shape != bias.theta_2d.shape:
            raise ValueError(f"initial_theta shape {theta.shape} does not match {bias.theta_2d.shape}")
    return theta

def _select_tridiag_solver(request: StaticRequest):
    if request.tridiag == "fast":
        from ..algorithms.thomas_fast import solve_const_offdiag_batched_fast
        return solve_const_offdiag_batched_fast
    return solve_const_offdiag_batched


def _target_power(beams) -> float:
    return float(sum(float(ch.power) for ch in beams.channels))


def _normalize_power(A, *, target_power: float, grid, coherent: bool, xp):
    I = total_intensity(A, coherent=coherent, xp=xp)
    p = xp.sum(I) * float(grid.dx_um) * float(grid.dy_um)
    scale = xp.sqrt(float(target_power) / (p + xp.asarray(1e-300, dtype=p.dtype)))
    return A * scale


def _field_difference(A, B, *, grid, xp) -> tuple[float, float]:
    """Return phase-aligned relative L2 difference and overlap magnitude."""
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    ov = xp.sum(xp.conj(A) * B) * dxdy
    pA = xp.sum(xp.abs(A) ** 2) * dxdy
    pB = xp.sum(xp.abs(B) ** 2) * dxdy
    ovn = ov / xp.sqrt(pA * pB + 1e-300)
    B_aligned = B * xp.exp(-1j * xp.angle(ovn))
    rel = xp.sqrt(xp.sum(xp.abs(B_aligned - A) ** 2) * dxdy / (pA + 1e-300))
    return float(asnumpy(rel)), float(asnumpy(xp.abs(ovn)))


def make_beta_symbol(grid, *, wavelength_um: float, n_ref: float, subtract_carrier: bool = True):
    """Angular-spectrum excess propagation symbol q(fx,fy)-q0."""
    xp = grid.xp
    q0 = 2.0 * math.pi * float(n_ref) / float(wavelength_um)
    arg = 1.0 - (float(wavelength_um) / float(n_ref)) ** 2 * grid.fxy2_um
    mask = arg > 0.0

    q = xp.zeros_like(grid.fxy2_um, dtype=grid.real_dtype)
    q[mask] = (q0 * xp.sqrt(arg[mask])).astype(grid.real_dtype, copy=False)

    if subtract_carrier:
        q = q - xp.asarray(q0, dtype=q.dtype)

    return q, mask


def optical_potential(theta, *, ne: float, no: float, n_ref: float, wavelength_um: float, xp):
    """V(theta)=k0*(n_eff(theta)-n_ref), matching nonlinear phase."""
    neff = neff_from_theta(theta, ne=ne, no=no, xp=xp)
    k0 = 2.0 * math.pi / float(wavelength_um)
    return k0 * (neff - float(n_ref))


def apply_optical_eigen_operator(A, theta, beta_symbol, *, grid, ne: float, no: float, n_ref: float, wavelength_um: float):
    """H(theta) A = q_perp A + V(theta) A."""
    xp = grid.xp
    Ahat = xp.fft.fft2(A)
    HA = xp.fft.ifft2(beta_symbol * Ahat)
    HA = HA + optical_potential(
        theta,
        ne=ne,
        no=no,
        n_ref=n_ref,
        wavelength_um=wavelength_um,
        xp=xp,
    ) * A
    return HA


def rayleigh_beta(A, theta, beta_symbol, *, grid, ne: float, no: float, n_ref: float, wavelength_um: float) -> float:
    """Rayleigh quotient beta = <A,H A>/<A,A>."""
    xp = grid.xp
    area = float(grid.dx_um) * float(grid.dy_um)
    HA = apply_optical_eigen_operator(
        A,
        theta,
        beta_symbol,
        grid=grid,
        ne=ne,
        no=no,
        n_ref=n_ref,
        wavelength_um=wavelength_um,
    )
    num = xp.vdot(A.ravel(), HA.ravel()) * area
    den = xp.vdot(A.ravel(), A.ravel()) * area
    return float(asnumpy(xp.real(num / den)))


def run_soliton(request: SolitonRequest) -> SolitonResult:
    """Run one clean soliton fixed-point solve."""

    request.validate()
    base = request.base

    backend = get_backend(base.backend)
    xp = backend.xp
    grid = make_grid(base.grid, xp=xp, real_dtype=backend.real_dtype)
    bias = build_bias(base.lc, grid)
    launch = build_launch(base.beams, grid, complex_dtype=backend.complex_dtype)
    tridiag_solver = _select_tridiag_solver(base)

    b = resolved_b(base.lc)
    bi = resolved_bi(base.lc, base.beams)
    target_power = _target_power(base.beams)
    coherent = launch.coherence == "coherent"

    A = _prepare_initial_A(
        request.initial_A,
        launch,
        grid=grid,
        target_power=target_power,
        coherent=coherent,
        xp=xp,
    )
    theta = _prepare_initial_theta(request.initial_theta, bias, xp=xp)

    n_ref = float(asnumpy(neff_from_theta(
        theta,
        ne=base.lc.material.ne,
        no=base.lc.material.no,
        xp=xp,
    )).mean())
    wavelength_um = float(asnumpy(launch.wavelengths_um[0]))

    n_ref_beta = float(asnumpy(neff_from_theta(
        bias.theta_2d,
        ne=base.lc.material.ne,
        no=base.lc.material.no,
        xp=xp,
    )).mean())

    dn_max_est = 0.02
    dz_opt_max_phi = 0.30
    max_substeps = 16
    phi_est = (2.0 * math.pi / wavelength_um) * float(grid.dz_um) * dn_max_est
    Nsub = max(1, min(max_substeps, int(math.ceil(phi_est / dz_opt_max_phi))))
    dz_sub = float(grid.dz_um) / Nsub
    h_sub = linear_kernel(grid.fxy2_um, dz=dz_sub, wavelength=wavelength_um, n_ref=n_ref, xp=xp)

    beta_symbol, beta_mask = make_beta_symbol(
        grid,
        wavelength_um=wavelength_um,
        n_ref=n_ref_beta,
        subtract_carrier=True,
     )


    s_cn, off_cn, diag_cn, _ = prepare_cn_operator(
        dt=7.5e-4,
        mobility=base.lc.mobility,
        dx=grid.du,
        dy=grid.dv,
        Ny=grid.Ny,
        xp=xp,
        dtype=backend.real_dtype,
    )

    def relax_theta(theta_in, intensity):
        out = theta_in
        for _ in range(int(request.theta_steps_per_outer)):
            out_new = cn_trapezoid_picard_step(
                out,
                intensity,
                intensity,
                b=b,
                bi=bi,
                dt=7.5e-4,
                mobility=base.lc.mobility,
                dx=grid.du,
                dy=grid.dv,
                s=s_cn,
                off=off_cn,
                diag=diag_cn,
                max_iter=4,
                tol_update=1e-6,
                clamp=bias.theta_clamp,
                tridiag_solver=tridiag_solver,
                xp=xp,
            )
            out = (1.0 - float(request.theta_mix)) * out + float(request.theta_mix) * out_new
            out[0, :] = base.lc.cell.theta_bc
            out[-1, :] = base.lc.cell.theta_bc
        return out

    def propagate(A_in, theta_in):
        A_out = A_in.copy()
        for _k in range(int(grid.Nz)):
            advance_slice(
                A_out,
                theta_in,
                kernel=h_sub,
                dz=float(grid.dz_um),
                wavelength=wavelength_um,
                n_ref=n_ref,
                ne=base.lc.material.ne,
                no=base.lc.material.no,
                Nsub=Nsub,
                xp=xp,
            )
        return A_out

    history: list[dict] = []
    converged = False

    synchronize(xp)
    t0 = _time.perf_counter()

    A_end = A.copy()
    intensity = total_intensity(A, coherent=coherent, xp=xp)

    for outer in range(int(request.max_outer)):
        A_prev = A.copy()
        theta_prev = theta.copy()

        intensity = total_intensity(A, coherent=coherent, xp=xp)
        theta = relax_theta(theta, intensity)

        A_end = propagate(A, theta)
        A_end = _normalize_power(A_end, target_power=target_power, grid=grid, coherent=coherent, xp=xp)

        # Phase align A_end to A before mixing.
        dxdy = float(grid.dx_um) * float(grid.dy_um)
        ov = xp.sum(xp.conj(A) * A_end) * dxdy
        phase = xp.angle(ov)
        A_candidate = A_end * xp.exp(-1j * phase)

        A = (1.0 - float(request.field_mix)) * A + float(request.field_mix) * A_candidate
        A = _normalize_power(A, target_power=target_power, grid=grid, coherent=coherent, xp=xp)

        field_rel, overlap_abs = _field_difference(A_prev, A, grid=grid, xp=xp)
        dtheta = theta - theta_prev
        dtheta_rms = float(asnumpy(xp.sqrt(xp.mean(dtheta * dtheta))))
        dtheta_max = float(asnumpy(xp.max(xp.abs(dtheta))))

        intensity = total_intensity(A, coherent=coherent, xp=xp)
        resid = residual_theta_static(
            theta,
            intensity,
            b=b,
            bi=bi,
            dx=grid.du,
            dy=grid.dv,
            theta_bc=base.lc.cell.theta_bc,
            xp=xp,
        )

        beta = {
            "beta": rayleigh_beta(
                A,
                theta,
                beta_symbol,
                grid=grid,
                ne=base.lc.material.ne,
                no=base.lc.material.no,
                n_ref=n_ref_beta,
                wavelength_um=wavelength_um,
            )
        }

        row = {
            "outer": int(outer + 1),
            "field_rel": field_rel,
            "overlap_abs": overlap_abs,
            "dtheta_rms": dtheta_rms,
            "dtheta_max": dtheta_max,
            **resid,
            **beta,
        }
        row.update(intensity_metrics(intensity, grid))
        row["theta_max"] = float(asnumpy(xp.max(theta)))
        history.append(row)

        if (
            field_rel < float(request.tol_field)
            and dtheta_rms < float(request.tol_theta)
            and resid["residual_rms"] < float(request.tol_residual_rms)
            and resid["residual_max"] < float(request.tol_residual_max)
        ):
            converged = True
            break

    synchronize(xp)
    elapsed = _time.perf_counter() - t0

    intensity = total_intensity(A, coherent=coherent, xp=xp)
    metrics = intensity_metrics(intensity, grid)
    metrics.update(residual_theta_static(
        theta,
        intensity,
        b=b,
        bi=bi,
        dx=grid.du,
        dy=grid.dv,
        theta_bc=base.lc.cell.theta_bc,
        xp=xp,
    ))
    if history:
        for key in (
            "field_rel",
            "overlap_abs",
            "dtheta_rms",
            "dtheta_max",
            "beta",
        ):
            if key in history[-1]:
                metrics[key] = history[-1][key]

    metrics.update({
        "backend": backend.name,
        "precision": base.backend.precision,
        "tridiag": base.tridiag,
        "Nx": grid.Nx,
        "Ny": grid.Ny,
        "Nz": grid.Nz,
        "outer_steps": len(history),
        "theta_steps_per_outer": int(request.theta_steps_per_outer),
        "field_mix": float(request.field_mix),
        "theta_mix": float(request.theta_mix),
        "converged": bool(converged),
        "target_power": float(target_power),
        "b": float(b),
        "bi": float(bi),
        "n_ref": float(n_ref),
        "Nsub": int(Nsub),
        "elapsed_s": float(elapsed),
        "theta_max": float(asnumpy(xp.max(theta))),
        "used_initial_A": bool(request.initial_A is not None),
        "used_initial_theta": bool(request.initial_theta is not None),
    })

    return SolitonResult(
        kind="SolitonResult",
        metrics=metrics,
        samples=history,
        A=A,
        theta=theta,
        intensity=intensity,
        history=history,
        converged=converged,
    )


__all__ = [
    "SolitonRequest",
    "SolitonResult",
    "run_soliton",
]
