from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cupy as cp
import cupyx.scipy.fft as spfft

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


from lc_offload_tools import (
    choose_optics_substeps,
    get_h_for_dz,
    prepare_ie_ky_operator,
    prepare_cn_ky_operator,
    hop_linear_inplace,
    intens_into,
    td_get_slice_mid_intensity,
    td_update_theta_from_midintensity,
    _slice_residual_rms_local3d_z,
)


@dataclass
class SplitTDResult:
    theta_full: object
    I_mid_store: object
    stores: object
    summary: dict


def _scalar_float(x):
    return float(x.get()) if hasattr(x, "get") else float(x)


def _ensure_td_defaults(ctx):
    if not hasattr(ctx, "_h_cache"):
        ctx._h_cache = {}

    defaults = dict(
        dn_max_est=0.02,
        dz_opt_max_phi=0.5,
        max_substeps=8,
        theta_z_gamma=0.0,
        theta_clamp=None,
        picard_iters=4,
        picard_tol_up=1e-6,
        true_td_full_pred_optics=False,
        runtime_every_k=0,
    )

    for k, v in defaults.items():
        if not hasattr(ctx, k):
            setattr(ctx, k, v)


def summarize_split_td_result(ctx, stores, I_mid_store):
    return dict(
        mode="true_td_frozen_I_split",
        Nt=int(len(stores.time_steps)),
        dt_first=_scalar_float(stores.time_steps[0]) if len(stores.time_steps) else None,
        t_total=float(cp.sum(stores.time_steps).get()) if hasattr(stores.time_steps, "get") else float(np.sum(stores.time_steps)),
        Nz=int(ctx.Nz),
        dz_um=float(ctx.dz),
        theta_shape=list(ctx.theta_full.shape),
        theta_min=float(cp.min(ctx.theta_full).get()),
        theta_max=float(cp.max(ctx.theta_full).get()),
        theta_mean=float(cp.mean(ctx.theta_full).get()),
        has_I_mid_store=I_mid_store is not None,
        has_Ixz=getattr(stores, "Ixz", None) is not None,
        has_Iyz=getattr(stores, "Iyz", None) is not None,
        has_thetaxz=getattr(stores, "thetaxz", None) is not None,
        has_thetayz=getattr(stores, "thetayz", None) is not None,
        I_mid_shape=list(I_mid_store.shape) if I_mid_store is not None else None,
        I_mid_min=float(cp.min(I_mid_store).get()) if I_mid_store is not None else None,
        I_mid_max=float(cp.max(I_mid_store).get()) if I_mid_store is not None else None,
        I_mid_mean=float(cp.mean(I_mid_store).get()) if I_mid_store is not None else None,
    )


def compute_frozen_I_mid_stack(
    ctx,
    theta_ref,
    *,
    store_dtype=cp.float32,
):
    """
    Pass 1 of split TD:
    march optics through all z using theta_ref only,
    and store I_mid[k].
    """
    _ensure_td_defaults(ctx)

    Nsub, dz_sub, phi_est = choose_optics_substeps(
        float(ctx.dz),
        kout=float(ctx.kout),
        dn_max_est=float(ctx.dn_max_est),
        dz_opt_max_phi=float(ctx.dz_opt_max_phi),
        max_substeps=int(ctx.max_substeps),
    )

    h_sub = get_h_for_dz(ctx, dz_sub)
    h_half = get_h_for_dz(ctx, 0.5 * dz_sub)

    ctx.Nsub = int(Nsub)
    ctx.dz_sub = float(dz_sub)

    amp = cp.empty_like(ctx.amp0)
    Ahat = cp.empty_like(ctx.amp0)

    I_b = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_a = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_mid = cp.empty((ctx.Nx, ctx.Ny), cp.float32)

    I_stack = cp.empty((ctx.Nz, ctx.Nx, ctx.Ny), dtype=store_dtype)

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    amp[...] = ctx.amp0

    # match old midpoint convention: half-hop before first slice
    hop_linear_inplace(
        amp,
        h_half,
        ctx.windowxy,
        Ahat,
        plan_f=plan_f,
        plan_i=plan_i,
    )

    for k in range(ctx.Nz):
        intens_into(I_b, amp, coh=ctx.coh)

        td_get_slice_mid_intensity(
            ctx,
            amp,
            theta_ref[k],
            I_b,
            I_a,
            I_mid,
            Nsub=Nsub,
            dz_sub=dz_sub,
            h_sub=h_sub,
            Ahat=Ahat,
            plan_f=plan_f,
            plan_i=plan_i,
            frozen_I_mid_k=None,
        )

        I_stack[k] = I_mid.astype(store_dtype, copy=False)

    return I_stack


def update_theta_stack_from_frozen_I(
    ctx,
    theta_ref,
    I_stack,
    dt_j,
    *,
    true_td=True,
    report_runtime=False,
):
    """
    Pass 2 of split TD:
    update every theta[k] from frozen I_stack[k].
    """
    _ensure_td_defaults(ctx)

    if true_td:
        a_ie, off, diag, lam_y = prepare_ie_ky_operator(
            dt=float(dt_j),
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=int(ctx.Ny),
        )
        s = None
    else:
        s, off, diag, lam_y = prepare_cn_ky_operator(
            dt=float(dt_j),
            mobility=float(ctx.mobility),
            du=float(ctx.du),
            dv=float(ctx.dv),
            Ny=int(ctx.Ny),
        )
        a_ie = None

    # dummy buffers required by td_update_theta_from_midintensity signature
    Ahat = cp.empty_like(ctx.amp0)
    I_b = cp.empty((ctx.Nx, ctx.Ny), cp.float32)
    I_a = cp.empty((ctx.Nx, ctx.Ny), cp.float32)

    plan_f = spfft.get_fft_plan(Ahat, axes=(-2, -1))
    plan_i = spfft.get_fft_plan(Ahat, axes=(-2, -1), value_type="C2C")

    td_resid_last = 0.0
    td_resid_max = 0.0

    theta_new_stack = cp.empty_like(ctx.theta_full)

    for k in range(ctx.Nz):
        tp = theta_ref[k - 1] if k > 0 else theta_ref[k]
        tn = theta_ref[k + 1] if (k + 1) < ctx.Nz else theta_ref[k]

        I_mid = I_stack[k].astype(cp.float32, copy=False)

        theta_new_stack[k] = td_update_theta_from_midintensity(
            ctx,
            theta_ref[k],
            tp,
            tn,
            I_mid,
            dt_j=float(dt_j),
            true_td=bool(true_td),
            a_ie=a_ie,
            s=s,
            off=off,
            diag=diag,
            lam_y=lam_y,
            Nsub=getattr(ctx, "Nsub", 1),
            dz_sub=getattr(ctx, "dz_sub", float(ctx.dz)),
            h_sub=None,
            amp_for_pred=None,
            I_b=I_b,
            I_a=I_a,
            Ahat=Ahat,
            plan_f=plan_f,
            plan_i=plan_i,
        )

        if report_runtime:
            rloc = _slice_residual_rms_local3d_z(
                theta_new_stack[k],
                I_mid,
                tp,
                tn,
                ctx,
                float(getattr(ctx, "theta_z_gamma", 0.0)),
            )
            td_resid_last = rloc
            td_resid_max = max(td_resid_max, rloc)

    ctx.theta_full[...] = theta_new_stack

    return td_resid_last, td_resid_max


def run_td_frozen_I_split_bridge(
    ctx,
    stores,
    *,
    restart_theta=False,
    store_I_dtype=cp.float32,
    true_td=True,
    save_slices=True,
    report_runtime=True,
    tqdm_timedep=True,
):
    """
    Physically split TD:

        theta^n
          -> compute full frozen I_mid^n(x,y,z)
          -> update theta^(n+1) from frozen I_mid^n
          -> repeat
    """
    _ensure_td_defaults(ctx)

    if restart_theta:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    time_steps = stores.time_steps
    tsteps = len(time_steps)

    jt_iter = range(tsteps)

    if tqdm_timedep and tqdm is not None:
        jt_iter = tqdm(jt_iter, desc="LC TD frozen-I split", dynamic_ncols=True)

    I_last = None

    for jt in jt_iter:
        dt_j = _scalar_float(time_steps[jt])

        theta_ref = ctx.theta_full.copy()

        I_stack = compute_frozen_I_mid_stack(
            ctx,
            theta_ref,
            store_dtype=store_I_dtype,
        )

        td_resid_last, td_resid_max = update_theta_stack_from_frozen_I(
            ctx,
            theta_ref,
            I_stack,
            dt_j,
            true_td=true_td,
            report_runtime=report_runtime,
        )

        I_last = I_stack

        if save_slices and getattr(stores, "save_slices", None) is not None:
            t_stride = int(getattr(stores, "t_stride", 1))
            if (jt % t_stride) == 0:
                slot = jt // t_stride
                for k in range(ctx.Nz):
                    stores.save_slices(
                        slot,
                        k,
                        I_stack[k].astype(cp.float32, copy=False),
                        ctx.theta_full[k],
                    )

        if report_runtime and hasattr(jt_iter, "set_postfix"):
            jt_iter.set_postfix(
                rlast=f"{td_resid_last:.2e}",
                rmax=f"{td_resid_max:.2e}",
            )

    summary = summarize_split_td_result(ctx, stores, I_last)

    return SplitTDResult(
        theta_full=ctx.theta_full,
        I_mid_store=I_last,
        stores=stores,
        summary=summary,
    )


def run_td_frozen_I_split_heun_bridge(
    ctx,
    stores,
    *,
    restart_theta=False,
    store_I_dtype=cp.float32,
    true_td=True,
    save_slices=True,
    report_runtime=True,
    tqdm_timedep=True,
):
    """
    Second-order candidate: frozen-I split predictor/corrector.

    For each physical time step:
        theta_n
          -> I_n from theta_n
          -> theta_star = EulerSplit(theta_n, I_n, dt)
          -> I_star from theta_star
          -> theta_starstar = EulerSplit(theta_star, I_star, dt)
          -> theta_np1 = 0.5 * theta_n + 0.5 * theta_starstar

    This is a practical Heun/trapezoid-like candidate using the existing
    semi-implicit theta step as the base one-step map.
    """
    _ensure_td_defaults(ctx)

    if restart_theta:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    time_steps = stores.time_steps
    tsteps = len(time_steps)

    jt_iter = range(tsteps)
    if tqdm_timedep and tqdm is not None:
        jt_iter = tqdm(jt_iter, desc="LC TD frozen-I Heun", dynamic_ncols=True)

    I_last = None

    for jt in jt_iter:
        dt_j = _scalar_float(time_steps[jt])

        # -------------------------------
        # Predictor: theta_n -> theta_star
        # -------------------------------
        theta_n = ctx.theta_full.copy()

        I_n = compute_frozen_I_mid_stack(
            ctx,
            theta_n,
            store_dtype=store_I_dtype,
        )

        update_theta_stack_from_frozen_I(
            ctx,
            theta_n,
            I_n,
            dt_j,
            true_td=true_td,
            report_runtime=False,
        )

        theta_star = ctx.theta_full.copy()

        # -------------------------------------
        # Corrector probe: theta_star -> theta**
        # -------------------------------------
        I_star = compute_frozen_I_mid_stack(
            ctx,
            theta_star,
            store_dtype=store_I_dtype,
        )

        update_theta_stack_from_frozen_I(
            ctx,
            theta_star,
            I_star,
            dt_j,
            true_td=true_td,
            report_runtime=False,
        )

        theta_starstar = ctx.theta_full

        # -------------------------------------
        # Heun / trapezoid-like combination
        # -------------------------------------
        ctx.theta_full[...] = (
            cp.float32(0.5) * theta_n
            + cp.float32(0.5) * theta_starstar
        )

        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        

        # Correct x-boundaries per slice
        ctx.theta_full[:, 0, :] = theta_bc
        ctx.theta_full[:, -1, :] = theta_bc

        if getattr(ctx, "theta_clamp", None) is not None:
            ctx.theta_full[...] = cp.clip(
                ctx.theta_full,
                cp.float32(ctx.theta_clamp[0]),
                cp.float32(ctx.theta_clamp[1]),
            )

        # Recompute final frozen intensity for output/slices
        I_last = compute_frozen_I_mid_stack(
            ctx,
            ctx.theta_full,
            store_dtype=store_I_dtype,
        )

        td_resid_last = 0.0
        td_resid_max = 0.0

        if report_runtime:
            for k in range(ctx.Nz):
                tp = ctx.theta_full[k - 1] if k > 0 else ctx.theta_full[k]
                tn = ctx.theta_full[k + 1] if (k + 1) < ctx.Nz else ctx.theta_full[k]

                rloc = _slice_residual_rms_local3d_z(
                    ctx.theta_full[k],
                    I_last[k].astype(cp.float32, copy=False),
                    tp,
                    tn,
                    ctx,
                    float(getattr(ctx, "theta_z_gamma", 0.0)),
                )
                td_resid_last = rloc
                td_resid_max = max(td_resid_max, rloc)

        if save_slices and getattr(stores, "save_slices", None) is not None:
            t_stride = int(getattr(stores, "t_stride", 1))
            if (jt % t_stride) == 0:
                slot = jt // t_stride
                for k in range(ctx.Nz):
                    stores.save_slices(
                        slot,
                        k,
                        I_last[k].astype(cp.float32, copy=False),
                        ctx.theta_full[k],
                    )

        if report_runtime and hasattr(jt_iter, "set_postfix"):
            jt_iter.set_postfix(
                rlast=f"{td_resid_last:.2e}",
                rmax=f"{td_resid_max:.2e}",
            )

    summary = summarize_split_td_result(ctx, stores, I_last)
    summary["mode"] = "true_td_frozen_I_split_heun"

    return SplitTDResult(
        theta_full=ctx.theta_full,
        I_mid_store=I_last,
        stores=stores,
        summary=summary,
    )

def update_theta_stack_trap_from_endpoint_I(
    ctx,
    theta_n,
    I_n,
    I_star,
    dt_j,
    *,
    report_runtime=False,
):
    """
    Trapezoidal corrector for frozen-I split TD.

    Uses endpoint optical forcings I_n and I_star.
    The theta update itself uses the existing CN/trapezoidal Picard slice solver.

    This is intended as a physically defensible second-order candidate:
        optics quasistatic at beginning and predicted endpoint,
        director updated with trapezoidal forcing.
    """
    _ensure_td_defaults(ctx)

    s_cn, off_cn, diag_cn, lam_y_cn = prepare_cn_ky_operator(
        dt=float(dt_j),
        mobility=float(ctx.mobility),
        du=float(ctx.du),
        dv=float(ctx.dv),
        Ny=int(ctx.Ny),
    )

    from lc_offload_tools import advance_theta_timestep_cn_trap_picard_prepared

    theta_new_stack = cp.empty_like(ctx.theta_full)

    td_resid_last = 0.0
    td_resid_max = 0.0

    for k in range(ctx.Nz):
        I0 = I_n[k].astype(cp.float32, copy=False)
        I1 = I_star[k].astype(cp.float32, copy=False)

        theta_new_stack[k] = advance_theta_timestep_cn_trap_picard_prepared(
            theta_n[k],
            dt=float(dt_j),
            b=ctx.b,
            bi=ctx.bi,
            I_n=I0,
            I_pic=I1,
            mobility=ctx.mobility,
            du=ctx.du,
            dv=ctx.dv,
            s=s_cn,
            off=off_cn,
            diag=diag_cn,
            lam_y=lam_y_cn,
            max_iter=int(getattr(ctx, "picard_iters", 4)),
            tol_update=float(getattr(ctx, "picard_tol_up", 1e-6)),
            clamp=ctx.theta_clamp,
        )

        theta_bc = cp.float32(getattr(ctx, "theta_bc", 0.0))
        theta_new_stack[k, 0, :] = theta_bc
        theta_new_stack[k, -1, :] = theta_bc

        if report_runtime:
            tp = theta_n[k - 1] if k > 0 else theta_n[k]
            tn = theta_n[k + 1] if (k + 1) < ctx.Nz else theta_n[k]

            I_avg = cp.float32(0.5) * (I0 + I1)

            rloc = _slice_residual_rms_local3d_z(
                theta_new_stack[k],
                I_avg,
                tp,
                tn,
                ctx,
                float(getattr(ctx, "theta_z_gamma", 0.0)),
            )
            td_resid_last = rloc
            td_resid_max = max(td_resid_max, rloc)

    ctx.theta_full[...] = theta_new_stack

    return td_resid_last, td_resid_max

def run_td_frozen_I_split_trap_bridge(
    ctx,
    stores,
    *,
    restart_theta=False,
    store_I_dtype=cp.float32,
    true_td=True,
    save_slices=True,
    report_runtime=True,
    tqdm_timedep=True,
):
    """
    Second-order predictor-corrector frozen-I split TD.

    For each physical time step:

        θⁿ
          -> compute Iⁿ = I[θⁿ]

        predictor:
          -> θ* = first-order split step using Iⁿ

        endpoint optics:
          -> compute I* = I[θ*]

        corrector:
          -> θⁿ⁺¹ using CN/trapezoidal theta update with endpoint forcing
             represented by Iⁿ and I*.

    This is the preferred second-order candidate for paper-level validation.
    """
    _ensure_td_defaults(ctx)

    if restart_theta:
        ctx.theta_full[...] = ctx.theta_bias_2d[None, :, :]

    time_steps = stores.time_steps
    tsteps = len(time_steps)

    jt_iter = range(tsteps)

    if tqdm_timedep and tqdm is not None:
        jt_iter = tqdm(jt_iter, desc="LC TD frozen-I trap", dynamic_ncols=True)

    I_last = None

    for jt in jt_iter:
        dt_j = _scalar_float(time_steps[jt])

        theta_n = ctx.theta_full.copy()

        # 1. Beginning-of-step optics
        I_n = compute_frozen_I_mid_stack(
            ctx,
            theta_n,
            store_dtype=store_I_dtype,
        )

        # 2. Predictor: first-order split Euler
        update_theta_stack_from_frozen_I(
            ctx,
            theta_n,
            I_n,
            dt_j,
            true_td=True,
            report_runtime=False,
        )

        theta_star = ctx.theta_full.copy()

        # 3. Predicted endpoint optics
        I_star = compute_frozen_I_mid_stack(
            ctx,
            theta_star,
            store_dtype=store_I_dtype,
        )

        # 4. Corrector: trapezoidal/CN theta step from theta_n
        ctx.theta_full[...] = theta_n

        td_resid_last, td_resid_max = update_theta_stack_trap_from_endpoint_I(
            ctx,
            theta_n,
            I_n,
            I_star,
            dt_j,
            report_runtime=report_runtime,
        )

        # 5. Accepted final intensity for output
        I_last = compute_frozen_I_mid_stack(
            ctx,
            ctx.theta_full,
            store_dtype=store_I_dtype,
        )

        if save_slices and getattr(stores, "save_slices", None) is not None:
            t_stride = int(getattr(stores, "t_stride", 1))
            if (jt % t_stride) == 0:
                slot = jt // t_stride
                for k in range(ctx.Nz):
                    stores.save_slices(
                        slot,
                        k,
                        I_last[k].astype(cp.float32, copy=False),
                        ctx.theta_full[k],
                    )

        if report_runtime and hasattr(jt_iter, "set_postfix"):
            jt_iter.set_postfix(
                rlast=f"{td_resid_last:.2e}",
                rmax=f"{td_resid_max:.2e}",
            )

    summary = summarize_split_td_result(ctx, stores, I_last)
    summary["mode"] = "true_td_frozen_I_split_trap"

    return SplitTDResult(
        theta_full=ctx.theta_full,
        I_mid_store=I_last,
        stores=stores,
        summary=summary,
    )