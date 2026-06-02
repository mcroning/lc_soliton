from __future__ import annotations

from typing import Tuple

__all__ = [
    "restrict_block_mean",
    "prolong_repeat",
]

def restrict_block_mean(arr, factor: int, *, xp):
    Nx, Ny = arr.shape
    f = int(factor)
    return arr.reshape(Nx // f, f, Ny // f, f).mean(axis=(1, 3)).astype(xp.float32)


def prolong_repeat(arr_c, factor: int, *, target_shape: Tuple[int, int], xp):
    f = int(factor)
    arr = xp.repeat(xp.repeat(arr_c, f, axis=0), f, axis=1)
    return arr[: target_shape[0], : target_shape[1]].astype(xp.float32)
