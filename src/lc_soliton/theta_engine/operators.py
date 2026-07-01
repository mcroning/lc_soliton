"""
theta_engine/operators.py

Fundamental PDE operators for the liquid-crystal director equation.

These routines contain no solver logic and no hard-coded precision.
They operate using the dtype/backend of the supplied arrays.

Boundary conditions:
    x : Dirichlet
    y : periodic
"""

from __future__ import annotations


def laplacian_xy(theta, *, du, dv, xp):
    """
    2-D Laplacian with Dirichlet x and periodic y.

    Parameters
    ----------
    theta : xp.ndarray (Nx, Ny)
    du, dv : float
        Grid spacings.
    xp : numpy or cupy module

    Returns
    -------
    lap : array, same dtype as theta
    """

    du2 = du * du
    dv2 = dv * dv

    lap = xp.zeros_like(theta)

    yp = xp.roll(theta, -1, axis=1)
    ym = xp.roll(theta, +1, axis=1)

    lap[1:-1, :] = (
        (theta[2:, :] - 2 * theta[1:-1, :] + theta[:-2, :]) / du2
        +
        (yp[1:-1, :] - 2 * theta[1:-1, :] + ym[1:-1, :]) / dv2
    )

    return lap


def laplacian_z(theta_prev, theta, theta_next, *, dz, gamma_z):
    """
    Nearest-neighbour z Laplacian.

    Returns gamma_z * d²θ/dz².
    """

    dz2 = dz * dz

    return gamma_z * (theta_next - 2 * theta + theta_prev) / dz2


def director_drive(theta, intensity, *, b, bi, xp):
    """
    Optical/electric nonlinear drive.

        (b + bi I) sin(2 theta)
    """

    return (b + bi * intensity) * xp.sin(2 * theta)


def theta_operator(
    theta,
    intensity,
    *,
    du,
    dv,
    b,
    bi,
    xp,
):
    """
    Spatial operator

        L(theta) + N(theta)

    without z-coupling.
    """

    return (
        laplacian_xy(theta, du=du, dv=dv, xp=xp)
        +
        director_drive(theta, intensity, b=b, bi=bi, xp=xp)
    )


def theta_operator_z(
    theta_prev,
    theta,
    theta_next,
    intensity,
    *,
    du,
    dv,
    dz,
    gamma_z,
    b,
    bi,
    xp,
):
    """
    Full spatial operator

        ∇²θ + gamma_z θzz + (b+biI) sin(2θ)
    """

    return (
        laplacian_xy(theta, du=du, dv=dv, xp=xp)
        +
        laplacian_z(
            theta_prev,
            theta,
            theta_next,
            dz=dz,
            gamma_z=gamma_z,
        )
        +
        director_drive(
            theta,
            intensity,
            b=b,
            bi=bi,
            xp=xp,
        )
    )


def apply_theta_bc(theta, theta_bc):
    """
    Apply Dirichlet boundary conditions in x.

    Operates in place.
    """

    theta[0, :] = theta_bc
    theta[-1, :] = theta_bc

    return theta


def clip_theta(theta, theta_clamp, *, xp):
    """
    Apply optional physical clipping.

    Operates in place.
    """

    if theta_clamp is None:
        return theta


    theta[...] = xp.clip(
        theta,
        theta_clamp[0],
        theta_clamp[1],
    )

    return theta