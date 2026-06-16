from __future__ import annotations

import math
from lc_soliton.core.backend import xp_default as cp, asnumpy
import scipy.special as spspec
__all__ = [
    "compute_neff",
    "compute_n_bg_from_bias",
    "compute_b_from_voltage",
    "compute_bi_from_power",
    "reported_freedericksz_voltage",
    "build_theta_bias_IC",
    "build_theta_bias_IC_dirichlet_value",
    "build_theta_bias_2d_from_physical",
    "build_theta_bias_2d_from_params",
    "theta_bias_2d_from_params",
    "lc_b_from_voltage",
    "lc_bi_from_power",
    "freedericksz_voltage",
    "voltage_from_lc_b",
    "theta0_from_b_zero_bc",
    "b_from_theta0_zero_bc",
    "theta_bias_1d_exact_zero_bc",
]


EPS0 = 8.854e-12
C0 = 3.0e8

def theta0_from_b_zero_bc(b):
    b_c = math.pi**2 / 8.0
    if float(b) <= b_c:
        return 0.0

    two_b = 2.0 * float(b)

    def Ksq_minus_2b(m):
        Km = spspec.ellipk(m)
        return float(Km * Km - two_b)

    m_lo, m_hi = 1e-12, 1.0 - 1e-12

    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)
        if Ksq_minus_2b(m_lo) * Ksq_minus_2b(m_mid) <= 0.0:
            m_hi = m_mid
        else:
            m_lo = m_mid

        if abs(m_hi - m_lo) < 1e-14:
            break

    m = 0.5 * (m_lo + m_hi)
    return float(math.asin(math.sqrt(m)))

def b_from_theta0_zero_bc(theta0):
    s = math.sin(float(theta0))
    m = s * s
    m = min(max(m, 1e-12), 1.0 - 1e-12)
    Km = spspec.ellipk(m)
    return float(0.5 * Km * Km)

def voltage_from_lc_b(b, *, K=7e-12, De=13.0):
    if float(b) <= 0:
        return 0.0
    return float(math.sqrt(8.0 * float(K) * float(b) / (float(De) * EPS0)))

def compute_b_from_voltage(V_bias, *, K=7e-12, De=13.0):
    return float(float(De) * EPS0 * float(V_bias) ** 2 / (8.0 * float(K)))


def compute_bi_from_power(P_mW, *, d_um=75.0, K=7e-12, ne=1.7, no=1.5):
    d_m = float(d_um) * 1e-6
    P_W = float(P_mW) * 1e-3
    na2 = float(ne) ** 2 - float(no) ** 2
    return float(na2 * d_m**2 * 1e12 * P_W / (8.0 * C0 * float(K)))


def reported_freedericksz_voltage(*, K=7e-12, De=13.0):
    b_c = math.pi**2 / 8.0
    return float(math.sqrt(8.0 * float(K) * b_c / (float(De) * EPS0)))


def compute_neff(theta, *, ne: float, no: float, xp):
    ct = xp.cos(theta)
    st = xp.sin(theta)
    inv_neff2 = (ct * ct) / (no * no) + (st * st) / (ne * ne)
    return 1.0 / xp.sqrt(inv_neff2)


def compute_n_bg_from_bias(theta_bias, ne, no, xp_mod, reducer="median"):
    xp = xp_mod
    neff_bias = compute_neff(theta_bias, ne=ne, no=no, xp=xp)
    if reducer == "mean":
        return float(xp.mean(neff_bias))
    return float(xp.median(neff_bias))


def _lam_y_periodic_second_diff(Ny, dv, xp=cp):
    k = xp.arange(Ny, dtype=xp.float32)
    return (-4.0 * xp.sin(xp.pi * k / Ny) ** 2) / (dv * dv)


def prepare_ie_ky_operator(*, dt, mobility, du, dv, Ny):
    a_ie = float(dt) / float(mobility)
    lam_y = _lam_y_periodic_second_diff(Ny, dv, xp=cp).astype(cp.float32, copy=False)
    off = cp.float32((-a_ie) * (1.0 / (du * du)))
    diag = (
        1.0
        + (2.0 * a_ie) * (1.0 / (du * du))
        - a_ie * lam_y
    ).astype(cp.float32, copy=False)
    return cp.float32(a_ie), off, diag, lam_y


def thomas_batched_const_tridiag(a, bvec, c, d_hatB):
    """
    Pure CuPy batched Thomas solve for rows of d_hatB.

    Solves, for each batch row:
        a x[i-1] + bvec[i] x[i] + c x[i+1] = d[i]

    d_hatB shape: (B, n)
    bvec shape: (B,) or (n,) depending legacy use.

    Here diag varies with ky, so bvec is usually shape (Ny,), one value per batch.
    """
    d = d_hatB.astype(cp.complex64, copy=True)
    B, n = d.shape

    a = cp.asarray(a, dtype=cp.float32)
    c = cp.asarray(c, dtype=cp.float32)
    bvec = cp.asarray(bvec, dtype=cp.float32)

    if bvec.ndim == 1 and bvec.shape[0] == B:
        b = cp.broadcast_to(bvec[:, None], (B, n)).astype(cp.float32, copy=True)
    elif bvec.ndim == 1 and bvec.shape[0] == n:
        b = cp.broadcast_to(bvec[None, :], (B, n)).astype(cp.float32, copy=True)
    else:
        b = cp.broadcast_to(bvec, (B, n)).astype(cp.float32, copy=True)

    cpv = cp.empty((B, n), dtype=cp.complex64)
    dpv = cp.empty((B, n), dtype=cp.complex64)

    cpv[:, 0] = c / b[:, 0]
    dpv[:, 0] = d[:, 0] / b[:, 0]

    for i in range(1, n):
        denom = b[:, i] - a * cpv[:, i - 1]
        cpv[:, i] = c / denom if i < n - 1 else 0.0
        dpv[:, i] = (d[:, i] - a * dpv[:, i - 1]) / denom

    x = cp.empty_like(d)
    x[:, -1] = dpv[:, -1]

    for i in range(n - 2, -1, -1):
        x[:, i] = dpv[:, i] - cpv[:, i] * x[:, i + 1]

    return x


def _laplacian_dirichletx_periody(theta, du, dv):
    th = theta.astype(cp.float32, copy=False)
    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    du2 = cp.float32(float(du) * float(du))
    dv2 = cp.float32(float(dv) * float(dv))

    lap = cp.zeros_like(th)
    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        + (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )
    return lap


def lc_residual2d_dirichletx_periody(theta, Ixy, *, b, bi, du, dv, mobility=1.0):
    th = theta.astype(cp.float64, copy=False)
    I64 = Ixy.astype(cp.float64, copy=False)

    du2 = cp.float64(du) * cp.float64(du)
    dv2 = cp.float64(dv) * cp.float64(dv)

    th_yplus = cp.roll(th, -1, axis=1)
    th_yminus = cp.roll(th, +1, axis=1)

    lap = cp.zeros_like(th)
    lap[1:-1, :] = (
        (th[2:, :] - 2.0 * th[1:-1, :] + th[:-2, :]) / du2
        + (th_yplus[1:-1, :] - 2.0 * th[1:-1, :] + th_yminus[1:-1, :]) / dv2
    )

    drive = (cp.float64(b) + cp.float64(bi) * I64) * cp.sin(2.0 * th)
    R = (lap + drive) / cp.float64(mobility)
    R[0, :] = 0.0
    R[-1, :] = 0.0
    return R


def build_theta_bias_IC(
    Nx, Ny, Nz, b,
    *,
    eps_clip=1e-12,
    return_1d=False,
    return_2d=False,
    dtype=cp.float32,
    kick_eps=1e-4,
    kick_seed=1234,
):
    xp = cp
    b_c = xp.pi**2 / 8.0

    if b <= float(b_c):
        theta_1d = xp.zeros(Nx, dtype=dtype)
        if return_1d:
            return theta_1d

        theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)

        if float(kick_eps) > 0.0:
            rng = xp.random.default_rng(int(kick_seed))
            kick = (
                float(kick_eps)
                * rng.standard_normal((Nx, Ny), dtype=cp.float32)
            ).astype(dtype, copy=False)
            kick[0, :] = 0.0
            kick[-1, :] = 0.0
            theta_2d = (theta_2d + kick).astype(dtype, copy=False)

        theta_2d[0, :] = 0.0
        theta_2d[-1, :] = 0.0

        if return_2d:
            return theta_2d
        return xp.repeat(theta_2d[:, :, None], Nz, axis=2)

    u = xp.linspace(-1.0, 1.0, Nx, dtype=cp.float64)
    two_b = 2.0 * float(b)

    def Ksq_minus_2b(m):
        Km = spspec.ellipk(m)
        return float(Km * Km - two_b)

    m_lo, m_hi = 1e-12, 1.0 - 1e-12

    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)
        if Ksq_minus_2b(m_lo) * Ksq_minus_2b(m_mid) <= 0.0:
            m_hi = m_mid
        else:
            m_lo = m_mid
        if abs(m_hi - m_lo) < 1e-14:
            break

    m = 0.5 * (m_lo + m_hi)

    theta0 = xp.arcsin(xp.sqrt(m))
    arg = xp.sqrt(2.0 * b) * u
    _, cn_cpu, dn_cpu, _ = spspec.ellipj(asnumpy(arg), float(m))

    cn = xp.asarray(cn_cpu)
    dn = xp.asarray(dn_cpu)
    cd = cn / dn

    s = xp.sin(theta0) * cd
    s = xp.clip(s, -1.0 + eps_clip, 1.0 - eps_clip)

    theta_1d = xp.arcsin(s).astype(dtype, copy=False)

    if return_1d:
        return theta_1d

    theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)
    theta_2d[0, :] = 0.0
    theta_2d[-1, :] = 0.0

    if return_2d:
        return theta_2d
    return xp.repeat(theta_2d[:, :, None], Nz, axis=2)

def build_theta_bias_IC_dirichlet_value(
    Nx, Ny, Nz, b,
    *,
    theta_bc=0.0,
    du=None,
    dv=None,
    mobility=1.0,
    max_iter=None,
    tol_rms=None,
    tol_max=None,
    dtau=None,
    report_every=None,
    verbose=False,
    eps_clip=1e-12,
    return_1d=False,
    return_2d=False,
    dtype=cp.float32,
):
    xp = cp
    if abs(float(theta_bc)) < 1e-14:
        return build_theta_bias_IC(
            Nx, Ny, Nz, b,
            eps_clip=eps_clip,
            return_1d=return_1d,
            return_2d=return_2d,
            dtype=dtype,
            kick_eps=0.0,
        )
    xp = cp

    u = xp.linspace(-1.0, 1.0, Nx, dtype=cp.float64)
    two_b = 2.0 * float(b)

    sbc = float(np.sin(theta_bc))

    # ---------------------------------------
    # solve for modulus m
    # ---------------------------------------

    if abs(theta_bc) < 1e-14:

        def f(m):
            Km = spspec.ellipk(m)
            return Km * Km - two_b

    else:

        sqrt2b = np.sqrt(two_b)

        def f(m):
            sn, cn, dn, ph = spspec.ellipj(sqrt2b, m)
            cd = cn / dn
            return np.sqrt(m) * cd - sbc

    m_lo = 1e-12
    m_hi = 1.0 - 1e-12

    flo = f(m_lo)
    fhi = f(m_hi)

    if flo * fhi > 0:
        raise RuntimeError(
            f"Could not bracket bias modulus: "
            f"f({m_lo})={flo}, f({m_hi})={fhi}"
        )

    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)

        if f(m_lo) * f(m_mid) <= 0:
            m_hi = m_mid
        else:
            m_lo = m_mid

        if abs(m_hi - m_lo) < 1e-14:
            break

    m = 0.5 * (m_lo + m_hi)

    # ---------------------------------------
    # evaluate profile
    # ---------------------------------------

    theta0 = xp.arcsin(xp.sqrt(m))

    arg = xp.sqrt(two_b) * u

    arg_cpu = asnumpy(arg)

    _, cn_cpu, dn_cpu, _ = spspec.ellipj(arg_cpu, float(m))

    cn = xp.asarray(cn_cpu)
    dn = xp.asarray(dn_cpu)

    cd = cn / dn

    s = xp.sin(theta0) * cd

    s = xp.clip(
        s,
        -1.0 + eps_clip,
        1.0 - eps_clip,
    )

    theta_1d = xp.arcsin(s).astype(dtype, copy=False)

    if return_1d:
        return theta_1d

    theta_2d = xp.tile(theta_1d[:, None], (1, Ny)).astype(dtype, copy=False)

    theta_2d[0, :] = cp.float32(theta_bc)
    theta_2d[-1, :] = cp.float32(theta_bc)

    if return_2d:
        return theta_2d

    return xp.repeat(theta_2d[:, :, None], Nz, axis=2)




def build_theta_bias_2d_from_physical(
    *,
    Nx,
    Ny,
    xaper_um,
    yaper_um,
    V_bias,
    theta_bc,
    K=7e-12,
    De=13.0,
    ne=1.7,
    no=1.5,
    mobility=1.0,
    theta_bias_max_iter=2000,
    theta_bias_tol_rms=5e-3,
    theta_bias_tol_max=2e-2,
    theta_bias_report_every=1000,
    theta_bias_verbose=False,
    xp=cp,
):
    b = compute_b_from_voltage(V_bias, K=K, De=De)

    Nx = int(Nx)
    Ny = int(Ny)

    du = 2.0 / max(1, Nx - 1)
    dv = (2.0 * (float(yaper_um) / float(xaper_um))) / Ny

    if abs(float(theta_bc)) < 1e-14:
        theta_bias_2d = build_theta_bias_IC(
            Nx,
            Ny,
            1,
            b,
            kick_eps=0.01,
            eps_clip=1e-12,
            return_2d=True,
        ).astype(xp.float32, copy=False)

        theta_bias_2d[0, :] = xp.float32(0.0)
        theta_bias_2d[-1, :] = xp.float32(0.0)

    else:
        theta_bias_2d = build_theta_bias_IC_dirichlet_value(
            Nx,
            Ny,
            1,
            b,
            theta_bc=float(theta_bc),
            du=du,
            dv=dv,
            mobility=float(mobility),
            max_iter=int(theta_bias_max_iter),
            tol_rms=float(theta_bias_tol_rms),
            tol_max=float(theta_bias_tol_max),
            report_every=int(theta_bias_report_every),
            verbose=bool(theta_bias_verbose),
            return_2d=True,
            dtype=xp.float32,
        ).astype(xp.float32, copy=False)

        theta_bias_2d[0, :] = xp.float32(theta_bc)
        theta_bias_2d[-1, :] = xp.float32(theta_bc)

    edge0 = float(xp.mean(theta_bias_2d[0, :]))
    edge1 = float(xp.mean(theta_bias_2d[-1, :]))

    if not (
        abs(edge0 - float(theta_bc)) < 1e-6
        and abs(edge1 - float(theta_bc)) < 1e-6
    ):
        raise RuntimeError(
            f"theta_bias mismatch: edges ({edge0}, {edge1}) vs theta_bc={theta_bc}"
        )

    n_bg = compute_n_bg_from_bias(theta_bias_2d, ne, no, xp, reducer="median")

    return {
        "b": float(b),
        "theta_bias_2d": theta_bias_2d,
        "n_bg": float(n_bg),
        "refin": float(n_bg),
        "du": float(du),
        "dv": float(dv),
    }


def build_theta_bias_2d_from_params(params, xp=cp):
    b = float(getattr(params, "b", 0.0))

    Nx = int(params.Nx)
    Ny = int(params.Ny)

    du = 2.0 / max(1, Nx - 1)
    dv = (2.0 * (float(params.yaper_um) / float(params.xaper_um))) / Ny

    theta_bc = float(params.theta_bc)

    if abs(theta_bc) < 1e-14:
        theta_bias_2d = build_theta_bias_IC(
            Nx,
            Ny,
            1,
            b,
            kick_eps=0.01,
            eps_clip=1e-12,
            return_2d=True,
        ).astype(xp.float32, copy=False)

        theta_bias_2d[0, :] = xp.float32(0.0)
        theta_bias_2d[-1, :] = xp.float32(0.0)

    else:
        theta_bias_2d = build_theta_bias_IC_dirichlet_value(
            Nx,
            Ny,
            1,
            b,
            theta_bc=theta_bc,
            du=du,
            dv=dv,
            mobility=float(getattr(params, "mobility", 1.0)),
            max_iter=int(getattr(params, "theta_bias_max_iter", 2000)),
            tol_rms=float(getattr(params, "theta_bias_tol_rms", 5e-3)),
            tol_max=float(getattr(params, "theta_bias_tol_max", 2e-2)),
            report_every=int(getattr(params, "theta_bias_report_every", 1000)),
            verbose=bool(getattr(params, "theta_bias_verbose", False)),
            return_2d=True,
            dtype=xp.float32,
        ).astype(xp.float32, copy=False)

        theta_bias_2d[0, :] = xp.float32(theta_bc)
        theta_bias_2d[-1, :] = xp.float32(theta_bc)

    return theta_bias_2d
##legacy wrapper
def theta_bias_1d_exact_zero_bc(Nx, b, *, xp=cp, dtype=None):
    if dtype is None:
        dtype = xp.float32 if hasattr(xp, "float32") else np.float32

    th = build_theta_bias_IC(
        int(Nx),
        1,
        1,
        float(b),
        return_1d=True,
        dtype=cp.float32,
        kick_eps=0.0,
    )

    th[0] = 0.0
    th[-1] = 0.0

    if xp is np:
        out = asnumpy(th).astype(dtype, copy=False)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    return th.astype(dtype, copy=False)

theta_bias_2d_from_params = build_theta_bias_2d_from_params
lc_b_from_voltage = compute_b_from_voltage
lc_bi_from_power = compute_bi_from_power
freedericksz_voltage = reported_freedericksz_voltage
