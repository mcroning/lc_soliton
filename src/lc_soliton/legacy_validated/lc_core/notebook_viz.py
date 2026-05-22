from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def _to_numpy(arr):
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def animate_td_cross_section(
    ctx,
    result,
    *,
    plane="yz",
    field="I",
    interval=250,
    percentile=99.5,
):
    """
    Animate TD saved cross-section movies.

    plane:
        "yz" or "xz"

    field:
        "I" or "theta"
    """
    if not hasattr(result, "stores"):
        raise TypeError("animate_td_cross_section requires a TD result with result.stores")

    if field == "I" and plane == "yz":
        arr = result.stores.Iyz
        ylabel = "y (um)"
        coord = _to_numpy(ctx.y)
        title_prefix = "TD Iyz"
    elif field == "I" and plane == "xz":
        arr = result.stores.Ixz
        ylabel = "x (um)"
        coord = _to_numpy(ctx.x)
        title_prefix = "TD Ixz"
    elif field == "theta" and plane == "yz":
        arr = result.stores.thetayz
        ylabel = "y (um)"
        coord = _to_numpy(ctx.y)
        title_prefix = "TD Δtheta yz"
    elif field == "theta" and plane == "xz":
        arr = result.stores.thetaxz
        ylabel = "x (um)"
        coord = _to_numpy(ctx.x)
        title_prefix = "TD Δtheta xz"
    else:
        raise ValueError("plane must be 'yz' or 'xz'; field must be 'I' or 'theta'")

    if arr is None:
        raise ValueError(f"No stored movie array for field={field!r}, plane={plane!r}")

    A = _to_numpy(arr)

    z_um = np.arange(A.shape[1]) * float(ctx.dz)

    fig, ax = plt.subplots(figsize=(8, 4))

    if field == "theta":
        vmax = np.percentile(np.abs(A), percentile)
        vmax = max(float(vmax), 1e-12)
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin = 0.0
        vmax = np.percentile(A, percentile)
        vmax = max(float(vmax), 1e-12)
        cmap = None

    im = ax.imshow(
        A[0].T,
        origin="lower",
        extent=[z_um.min(), z_um.max(), coord.min(), coord.max()],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel("z (um)")
    ax.set_ylabel(ylabel)
    title = ax.set_title(f"{title_prefix} frame 0")
    fig.colorbar(im, ax=ax)

    def update(i):
        im.set_data(A[i].T)
        title.set_text(f"{title_prefix} frame {i}")
        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=A.shape[0],
        interval=interval,
        blit=False,
    )

    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_static_cross_section(
    ctx,
    result,
    *,
    plane="yz",
    field="I",
    percentile=99.5,
    frame=None,
):
    """
    Plot static z-cross-section.

    plane:
        "yz" or "xz"

    field:
        "I" or "theta"
    """
    if field == "I":
        stack = result.I_mid_store
    elif field == "theta":
        stack = result.theta_full - ctx.theta_bias_2d[None, :, :]
    else:
        raise ValueError("field must be 'I' or 'theta'")

    A3 = _to_numpy(stack)

    # If result is TD and frame is requested, use stored TD movie frames.
    if hasattr(result, "stores") and frame is not None:
        if field == "I" and plane == "yz":
            A = _to_numpy(result.stores.Iyz)[frame]
            coord = _to_numpy(ctx.y)
            ylabel = "y (um)"
            title = f"TD I yz frame {frame}"
        elif field == "I" and plane == "xz":
            A = _to_numpy(result.stores.Ixz)[frame]
            coord = _to_numpy(ctx.x)
            ylabel = "x (um)"
            title = f"TD I xz frame {frame}"
        elif field == "theta" and plane == "yz":
            A = _to_numpy(result.stores.thetayz)[frame]
            coord = _to_numpy(ctx.y)
            ylabel = "y (um)"
            title = f"TD Δtheta yz frame {frame}"
        elif field == "theta" and plane == "xz":
            A = _to_numpy(result.stores.thetaxz)[frame]
            coord = _to_numpy(ctx.x)
            ylabel = "x (um)"
            title = f"TD Δtheta xz frame {frame}"
        else:
            raise ValueError("plane must be 'yz' or 'xz'; field must be 'I' or 'theta'")
    
    else:
        if field == "I":
            stack = result.I_mid_store
        elif field == "theta":
            stack = result.theta_full - ctx.theta_bias_2d[None, :, :]
        else:
            raise ValueError("field must be 'I' or 'theta'")
    
        A3 = _to_numpy(stack)
    
        if plane == "yz":
            A = A3[:, ctx.Nx // 2, :]
            coord = _to_numpy(ctx.y)
            ylabel = "y (um)"
            title = f"Final {field} yz"
        elif plane == "xz":
            A = A3[:, :, ctx.Ny // 2]
            coord = _to_numpy(ctx.x)
            ylabel = "x (um)"
            title = f"Final {field} xz"
        else:
            raise ValueError("plane must be 'yz' or 'xz'")

    z_um = np.arange(A.shape[0]) * float(ctx.dz)

    fig, ax = plt.subplots(figsize=(8, 4))

    if field == "theta":
        vmax = np.percentile(np.abs(A), percentile)
        vmax = max(float(vmax), 1e-12)
        vmin = -vmax
        cmap = "RdBu_r"
    else:
        vmin = 0.0
        vmax = np.percentile(A, percentile)
        vmax = max(float(vmax), 1e-12)
        cmap = None

    im = ax.imshow(
        A.T,
        origin="lower",
        extent=[z_um.min(), z_um.max(), coord.min(), coord.max()],
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel("z (um)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    plt.show()
    return fig


def animate_z_slices(
    ctx,
    result,
    *,
    field="I",
    z_stride=10,
    interval=80,
    percentile=99.5,
):
    """
    Animate x-y slices through z from full I_mid_store or theta_full.
    """
    if field == "I":
        A3 = _to_numpy(result.I_mid_store)
        title_prefix = "I(x,y)"
        cmap = None
        vmin = 0.0
        vmax = np.percentile(A3, percentile)
        vmax = max(float(vmax), 1e-12)
    elif field == "theta":
        A3 = _to_numpy(result.theta_full - ctx.theta_bias_2d[None, :, :])
        title_prefix = "Δtheta(x,y)"
        cmap = "RdBu_r"
        vmax = np.percentile(np.abs(A3), percentile)
        vmax = max(float(vmax), 1e-12)
        vmin = -vmax
    else:
        raise ValueError("field must be 'I' or 'theta'")

    A = A3[::z_stride]

    x_um = _to_numpy(ctx.x)
    y_um = _to_numpy(ctx.y)

    fig, ax = plt.subplots(figsize=(5, 5))

    im = ax.imshow(
        A[0].T,
        origin="lower",
        extent=[x_um.min(), x_um.max(), y_um.min(), y_um.max()],
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    title = ax.set_title(f"{title_prefix}, z = 0.0 um")
    fig.colorbar(im, ax=ax)

    def update(i):
        im.set_data(A[i].T)
        z_here = i * z_stride * float(ctx.dz)
        title.set_text(f"{title_prefix}, z = {z_here:.1f} um")
        return im, title

    ani = FuncAnimation(
        fig,
        update,
        frames=A.shape[0],
        interval=interval,
        blit=False,
    )

    plt.close(fig)
    return HTML(ani.to_jshtml())