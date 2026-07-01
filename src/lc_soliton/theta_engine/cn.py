"""
theta_engine/cn.py

Crank-Nicolson linear algebra for the LC theta equation.

No hard-coded precision:
    arrays and dtype policy are chosen by the caller.

Boundary conditions:
    x : Dirichlet
    y : periodic
"""

from __future__ import annotations


def lam_y_periodic_second_diff(Ny, dv, *, xp, dtype=None):
    """
    Eigenvalues of periodic second-difference operator in y.
    """
    if dtype is None:
        k = xp.arange(Ny)
    else:
        k = xp.arange(Ny, dtype=dtype)

    return (-4 * xp.sin(xp.pi * k / Ny) ** 2) / (dv * dv)


def prepare_cn_operator(*, dt, mobility, du, dv, Ny, xp, dtype=None):
    """
    Build coefficients for

        (I - dt/(2M) Lxy)

    after FFT diagonalization in y.
    """
    s = dt / (2 * mobility)

    lam_y = lam_y_periodic_second_diff(
        Ny,
        dv,
        xp=xp,
        dtype=dtype,
    )

    off = -s / (du * du)
    diag = 1 + 2 * s / (du * du) - s * lam_y

    if dtype is not None:
        s = dtype.type(s) if hasattr(dtype, "type") else dtype(s)
        off = dtype.type(off) if hasattr(dtype, "type") else dtype(off)
        diag = diag.astype(dtype, copy=False)
        lam_y = lam_y.astype(dtype, copy=False)

    return s, off, diag, lam_y


def solve_tridiag_batched_const(off, diag, rhs_hat, *, theta_bc_hat=None, xp):
    """
    Solve batched tridiagonal systems along x for each y-Fourier mode.

    Matrix:
        diag[j] on the diagonal for Fourier mode j
        off on lower/upper diagonals

    rhs_hat shape: (Nx, Ny), complex.
    """
    Nx, Ny = rhs_hat.shape
    n = Nx - 2

    out = xp.zeros_like(rhs_hat)

    if n <= 0:
        out[...] = rhs_hat
        return out

    if theta_bc_hat is None:
        theta_bc_hat = xp.zeros((Ny,), dtype=rhs_hat.dtype)

    # Interior RHS, corrected for Dirichlet boundary rows.
    d = rhs_hat[1:-1, :].copy()
    d[0, :] -= off * theta_bc_hat
    d[-1, :] -= off * theta_bc_hat

    # Thomas forward coefficients, one vector per Fourier mode.
    cprime = xp.empty((n, Ny), dtype=rhs_hat.dtype)
    dprime = xp.empty((n, Ny), dtype=rhs_hat.dtype)

    denom = diag.astype(rhs_hat.dtype, copy=False)
    cprime[0, :] = off / denom
    dprime[0, :] = d[0, :] / denom

    for i in range(1, n):
        denom = diag.astype(rhs_hat.dtype, copy=False) - off * cprime[i - 1, :]
        cprime[i, :] = off / denom if i < n - 1 else 0
        dprime[i, :] = (d[i, :] - off * dprime[i - 1, :]) / denom

    x = xp.empty((n, Ny), dtype=rhs_hat.dtype)
    x[-1, :] = dprime[-1, :]

    for i in range(n - 2, -1, -1):
        x[i, :] = dprime[i, :] - cprime[i, :] * x[i + 1, :]

    out[0, :] = theta_bc_hat
    out[-1, :] = theta_bc_hat
    out[1:-1, :] = x

    return out


def solve_cn_linear_system(rhs, *, off, diag, theta_bc, xp):
    """
    Solve

        (I - s Lxy) theta = rhs

    using FFT in y and batched Thomas in x.
    """
    rhs2 = rhs.copy()
    rhs2[0, :] = theta_bc
    rhs2[-1, :] = theta_bc

    rhs_hat = xp.fft.fft(rhs2, axis=1)

    # Fourier transform of constant x-boundary row.
    bc_row = xp.empty((rhs.shape[1],), dtype=rhs.dtype)
    bc_row[...] = theta_bc
    theta_bc_hat = xp.fft.fft(bc_row)

    out_hat = solve_tridiag_batched_const(
        off,
        diag,
        rhs_hat,
        theta_bc_hat=theta_bc_hat,
        xp=xp,
    )

    theta = xp.fft.ifft(out_hat, axis=1).real
    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc

    return theta.astype(rhs.dtype, copy=False)