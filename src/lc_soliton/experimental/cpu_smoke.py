"""
Archived CPU smoke numerics.

These routines are preserved for future CPU validation work but are not used
by the validated engine. Do not import into production paths without tests.
"""

# -----------------------------------------------------------------------------
# Local package-smoke fallback numerics
# -----------------------------------------------------------------------------


def _fft2(xp, a, axes=(-2, -1)):
    return _cupy_fft.fft2(a, axes=axes) if (_HAS_CUPY and xp is _cupy) else np.fft.fft2(a, axes=axes)


def _ifft2(xp, a, axes=(-2, -1)):
    return _cupy_fft.ifft2(a, axes=axes) if (_HAS_CUPY and xp is _cupy) else np.fft.ifft2(a, axes=axes)


def _compute_h_local(ctx: LCContext, dz_um: float):
    key = round(float(dz_um), 12)
    if key in ctx._h_cache:
        return ctx._h_cache[key]
    xp = ctx.xp
    lm = float(ctx.p.wavelength_um)
    refin = float(ctx.refin)
    arg = 1.0 - (lm / refin) ** 2 * ctx.fxy2
    h = xp.where(arg > 0, xp.exp(2j * xp.pi * refin * dz_um / lm * xp.sqrt(xp.maximum(arg, 0))), 0)
    ctx._h_cache[key] = h.astype(xp.complex64)
    return ctx._h_cache[key]


def _hop_linear_local(ctx: LCContext, amp, h):
    xp = ctx.xp
    A = _fft2(xp, amp, axes=(-2, -1))
    A *= h
    out = _ifft2(xp, A, axes=(-2, -1)).astype(xp.complex64)
    if ctx.windowxy is not None:
        out *= ctx.windowxy[None, :, :]
    return out


def _lc_dn_local(theta, ctx: LCContext):
    return (compute_neff(theta, ne=ctx.p.ne, no=ctx.p.no, xp=ctx.xp) - ctx.refin).astype(ctx.xp.float32)


def _propagate_slice_local(ctx: LCContext, amp, theta, *, nsub: int, dz_sub: float, h_sub):
    xp = ctx.xp
    for _ in range(int(nsub)):
        phase = xp.exp((1j * xp.float32(ctx.kout * dz_sub)) * _lc_dn_local(theta, ctx)).astype(xp.complex64)
        amp = (amp * phase[None, :, :]).astype(xp.complex64)
        amp = _hop_linear_local(ctx, amp, h_sub)
    return amp


def _laplacian_local(theta, du: float, dv: float, xp):
    th = theta.astype(xp.float32, copy=False)
    lap = xp.zeros_like(th)
    yp = xp.roll(th, -1, axis=1)
    ym = xp.roll(th, 1, axis=1)
    lap[1:-1, :] = ((th[2:, :] - 2 * th[1:-1, :] + th[:-2, :]) / (du * du) + (yp[1:-1, :] - 2 * th[1:-1, :] + ym[1:-1, :]) / (dv * dv))
    return lap


def _residual_local(theta, I, ctx: LCContext):
    p = ctx.p
    xp = ctx.xp
    R = (_laplacian_local(theta, ctx.du_lc, ctx.dv_lc, xp) + (p.b + p.bi * I) * xp.sin(2 * theta)) / p.mobility
    R[0, :] = 0
    R[-1, :] = 0
    return R.astype(xp.float32)


def _simple_static_smoke_slice(ctx: LCContext, amp, theta_seed, *, nsub: int, dz_sub: float, h_sub):
    """Small CPU-safe smoke path. Not the trusted physics branch."""
    xp = ctx.xp
    p = ctx.p
    I_b = intensity(amp, p.coherent, xp=xp)
    theta = theta_seed.astype(xp.float32, copy=True)
    theta[0, :] = xp.float32(p.theta_bc)
    theta[-1, :] = xp.float32(p.theta_bc)

    # Use a very conservative explicit relaxation. This exists only so CPU tests
    # can exercise IO/request plumbing; trusted static uses validated_core on GPU.
    dtau = min(float(p.dtau_static), 1e-5)
    for _ in range(max(1, min(int(p.static_max_steps), 20))):
        R = _residual_local(theta, I_b, ctx)
        theta[1:-1, :] += xp.float32(dtau) * R[1:-1, :]
        theta = xp.clip(theta, p.theta_clamp_min, p.theta_clamp_max).astype(xp.float32)
        theta[0, :] = xp.float32(p.theta_bc)
        theta[-1, :] = xp.float32(p.theta_bc)

    amp_out = _propagate_slice_local(ctx, amp.copy(), theta, nsub=nsub, dz_sub=dz_sub, h_sub=h_sub)
    I_a = intensity(amp_out, p.coherent, xp=xp)
    I_mid = 0.5 * (I_b + I_a)
    R = _residual_local(theta, I_mid, ctx)
    Rint = R[1:-1, :]
    info = dict(
        rrms=float(asnumpy(xp.sqrt(xp.mean(Rint * Rint)))),
        rmax=float(asnumpy(xp.max(xp.abs(Rint)))),
        converged=True,
        method="package_smoke",
    )
    return theta, I_mid, amp_out, info


def _make_scalar_info_from_residual(theta, I_mid, ctx: LCContext) -> Dict[str, Any]:
    R = lc_residual64(theta, I_mid, b=ctx.b, bi=ctx.bi, du=ctx.du, dv=ctx.dv, mobility=ctx.mobility)
    stats = residual_stats_2d(R)
    return normalize_info(dict(rrms=stats["rms_interior"], rmax=stats["max_interior"], converged=True))
