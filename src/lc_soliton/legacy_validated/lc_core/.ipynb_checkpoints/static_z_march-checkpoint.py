# lc_core/static_z_march.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .residuals import lc_residual_slice3d_ctx, residual_stats_2d
from .optics import intensity, total_power


@dataclass
class StaticZMarchResult:
    theta_full: Any
    I_mid_store: Any
    final_amp: Any
    reports: list[dict]
    summary: dict


def _ensure_legacy_ctx_defaults(ctx):
    defaults = dict(
        theta_clamp=None,
        dn_max_est=0.02,
        dz_opt_max_phi=0.5,
        max_substeps=8,
        strict_selfcons_passes=3,
        strict_selfcons_tol_theta=1e-4,
        strict_selfcons_tol_I=1e-4,
        runtime_every_k=0,
    )

    for k, v in defaults.items():
        if not hasattr(ctx, k):
            setattr(ctx, k, v)

    if not hasattr(ctx, "_h_cache"):
        ctx._h_cache = {}

    if not hasattr(ctx, "h_cache"):
        ctx.h_cache = ctx._h_cache


def run_static_strict_z_march_bridge(
    ctx,
    *,
    choose_optics_substeps,
    get_h_for_dz_legacy,
    prepare_cn_ky_operator,
    strict_static_relax_slice_selfconsistent,
    restart_static_from_bias: bool = True,
    use_linear_seed_for_static_relax: bool = True,
    early_accept_linear_seed: bool = True,
    strict_residual_tol_max: float = 6e-2,
    strict_residual_tol_rms: float = 1e-2,
    strict_max_outer_passes: int = 8,
    store_I_dtype=None,
    verbose_every: int | None = 50,
    tqdm_z: bool = True,
):
    """
    Strict static z-march bridge.

    This is intentionally a controlled wrapper around the validated legacy
    self-consistent strict slice kernel:

        strict_static_relax_slice_selfconsistent(...)

    The new module owns:
        - z march order
        - ctx defaults
        - report list
        - I_mid_store allocation
        - residual summaries

    Parameters
    ----------
    ctx:
        New LCContextGPU.
    choose_optics_substeps, get_h_for_dz_legacy, prepare_cn_ky_operator,
    strict_static_relax_slice_selfconsistent:
        Legacy validated functions injected explicitly.

    Returns
    -------
    StaticZMarchResult
    """


    # =========================================================
    # OPTIONAL X-PARITY SYMMETRIZATION DEBUG HELPERS
    # =========================================================
    
    ENFORCE_X_SYMMETRY_DEBUG = False
    
    def sym_x(A):
        """
        Symmetrize a real-valued x-y array over x.
        Shape: (Nx, Ny)
        """
        return 0.5 * (A + A[::-1, :])
    
    def sym_x_stack(A):
        """
        Symmetrize a real-valued z-x-y stack over x.
        Shape: (Nz, Nx, Ny)
        """
        return 0.5 * (A + A[:, ::-1, :])
    
    def sym_x_amp(A):
        """
        Symmetrize optical field array over x.
        Shape: (nbeam, Nx, Ny)
        """
        return 0.5 * (A + A[:, ::-1, :])
    
    def apply_symmetry_debug_patch(
        *,
        theta=None,
        I=None,
        amp=None,
    ):
        """
        Apply optional x-parity symmetrization.
    
        Returns
        -------
        theta_out, I_out, amp_out
        """
        if not ENFORCE_X_SYMMETRY_DEBUG:
            return theta, I, amp
    
        theta_out = theta
        I_out = I
        amp_out = amp
    
        if theta is not None:
            theta_out = sym_x(theta)
    
        if I is not None:
            I_out = sym_x(I)
    
        if amp is not None:
            amp_out = sym_x_amp(amp)
    
        return theta_out, I_out, amp_out
###################




    
    cp = ctx.xp
    _ensure_legacy_ctx_defaults(ctx)

    if store_I_dtype is None:
        store_I_dtype = cp.float32

    if restart_static_from_bias:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    
    # Legacy optics substep selection.
    Nsub, dz_sub, phi_est = choose_optics_substeps(
        float(ctx.dz),
        kout=float(ctx.kout),
        dn_max_est=float(getattr(ctx, "dn_max_est", 0.02)),
        dz_opt_max_phi=float(getattr(ctx, "dz_opt_max_phi", 0.5)),
        max_substeps=int(getattr(ctx, "max_substeps", 8)),
    )

    h_sub = get_h_for_dz_legacy(ctx, dz_sub)

    ctx.Nsub = int(Nsub)
    ctx.dz_sub = float(dz_sub)

    # Buffers and FFT plans for legacy hop_linear_inplace.
    try:
        import cupyx.scipy.fft as spfft
        Ahat = cp.empty_like(ctx.amp0)
        plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
        plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")
    except Exception:
        # Many legacy functions tolerate None plans.
        Ahat = cp.empty_like(ctx.amp0)
        plan_f = None
        plan_i = None

    sS, offS, diagS, lamS = prepare_cn_ky_operator(
        dt=float(ctx.dtau_static),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
    )

    amp = ctx.amp0.copy()
    I_mid_store = cp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=store_I_dtype)

    reports = []

    iterator = range(int(ctx.Nz))
    if tqdm_z:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, desc="strict static z", dynamic_ncols=True)
        except Exception:
            pass

    theta_running = ctx.theta_bias_2d.copy()

    print(
        f"[strict z bridge] Nsub={Nsub} dz_sub={dz_sub:.6g} "
        f"phi_est={phi_est:.3g} Nz={ctx.Nz}"
    )

    for k in iterator:
        # z-neighbor convention. For gamma_z=0 this is irrelevant; for gamma_z>0
        # it gives a lagged/predicted local residual structure.
        if k == 0:
            tp = ctx.theta_full[0]
        else:
            tp = ctx.theta_full[k - 1]

        if k + 1 < ctx.Nz:
            tn = ctx.theta_full[k + 1]
        else:
            tn = ctx.theta_full[k]

        theta_seed = (
            ctx.theta_bias_2d
            if restart_static_from_bias and k == 0
            else theta_running
        )

        theta_running, I_mid_sc, amp_sc, info = strict_static_relax_slice_selfconsistent(
            amp_in=amp,
            theta_seed=theta_seed,
            tp=tp,
            tn=tn,
            ctx=ctx,
            dz_sub=float(dz_sub),
            Nsub=int(Nsub),
            h_sub=h_sub,
            Ahat=Ahat,
            plan_f=plan_f,
            plan_i=plan_i,
            sS=sS,
            offS=offS,
            diagS=diagS,
            lamS=lamS,
            use_linear_seed=bool(use_linear_seed_for_static_relax),
            early_accept_linear_seed=bool(early_accept_linear_seed),
            residual_tol_max=float(strict_residual_tol_max),
            residual_tol_rms=float(strict_residual_tol_rms),
            max_outer_passes=int(strict_max_outer_passes),
            max_selfcons_passes=int(getattr(ctx, "strict_selfcons_passes", 3)),
            selfcons_tol_theta=float(getattr(ctx, "strict_selfcons_tol_theta", 1e-4)),
            selfcons_tol_I=float(getattr(ctx, "strict_selfcons_tol_I", 1e-4)),
            verbose=False,
        )


        theta_running, I_mid_sc, amp_sc = apply_symmetry_debug_patch(        
            theta=theta_running,        
            I=I_mid_sc,        
            amp=amp_sc,        
        )
        
        ctx.theta_full[k] = theta_running
        if False:
            if k in (0, 1, 10, ctx.Nz - 1):
                # after strict_static_relax_slice_selfconsistent returns
                odd_theta = theta_running - theta_running[::-1, :]
                odd_I = I_mid_sc - I_mid_sc[::-1, :]
                
                print(
                    "[strict return odd]",
                    "theta=", float(cp.max(cp.abs(odd_theta)).get()),
                    "I=", float(cp.max(cp.abs(odd_I)).get()),
                )
        
        
        #### temp
                ctx.theta_full[k] = 0.5 * (
                    ctx.theta_full[k]
                    + ctx.theta_full[k][::-1, :]
                )
        
                if k in (0, 1, 10, ctx.Nz - 1):
                    odd = ctx.theta_full[k] - ctx.theta_full[k][::-1, :]
                    print(
                        f"[sym debug k={k}] "
                        f"theta odd max after sym = {float(cp.max(cp.abs(odd)).get()):.3e}"
                    )
        ####temp
        if ENFORCE_X_SYMMETRY_DEBUG:
            I_mid_sc = 0.5 * (
                I_mid_sc
                + I_mid_sc[::-1, :]
            )
        
        I_mid_store[k] = I_mid_sc.astype(store_I_dtype, copy=False)

        
        amp[...] = amp_sc

        # Common new-core local residual report for this solved slice.
        Rloc = lc_residual_slice3d_ctx(ctx, theta_running, I_mid_sc, tp, tn)
        rstats = residual_stats_2d(Rloc)

        row = dict(info)
        row.update(
            dict(
                k=int(k),
                z_um=float(k * ctx.dz),
                residual_new_core=rstats,
                max_interior_new_core=rstats["max_interior"],
                rms_interior_new_core=rstats["rms_interior"],
            )
        )
        reports.append(row)

        if verbose_every is not None and (
            k == 0 or k == ctx.Nz - 1 or (int(verbose_every) > 0 and k % int(verbose_every) == 0)
        ):
            print(
                f"[k={k:4d}] "
                f"conv={row.get('converged')} "
                f"self={row.get('n_selfcons_passes')} "
                f"max={rstats['max_interior']:.3e} "
                f"rms={rstats['rms_interior']:.3e}"
            )

    Ifinal = intensity(amp, coh=ctx.coh).astype(cp.float32, copy=False)
    Pfinal = total_power(Ifinal, ctx.dx, ctx.dy)

    summary = dict(
        mode="strict_static_z_march_bridge",
        Nz=int(ctx.Nz),
        dz_um=float(ctx.dz),
        Nsub=int(Nsub),
        dz_sub_um=float(dz_sub),
        phi_est=float(phi_est),
        converged_count=int(sum(bool(r.get("converged", False)) for r in reports)),
        failed_count=int(sum(not bool(r.get("converged", False)) for r in reports)),
        max_residual_max=float(max(r["max_interior_new_core"] for r in reports)) if reports else None,
        max_residual_rms=float(max(r["rms_interior_new_core"] for r in reports)) if reports else None,
        P_final=float(Pfinal),
    )

    return StaticZMarchResult(
        theta_full=ctx.theta_full,
        I_mid_store=I_mid_store,
        final_amp=amp,
        reports=reports,
        summary=summary,
    )


def write_static_z_march_outputs(result: StaticZMarchResult, run_dir):
    """
    Save JSON summary and CSV-like reports as JSON.
    Large arrays are not saved here by default.
    """
    import json
    from pathlib import Path

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "static_z_summary.json").write_text(json.dumps(result.summary, indent=2))
    (run_dir / "static_z_reports.json").write_text(json.dumps(result.reports, indent=2))

    return result.summary
