"""
Example: load an existing LC run directory.
"""

from pathlib import Path

from lc_soliton import load_run

RUN_DIR = Path("PATH_TO_RUN_DIRECTORY")


def main():
    print(f"Loading run from: {RUN_DIR}")

    ctx, prdata = load_run(RUN_DIR)

    print("Run loaded successfully.")
    print(f"Nx = {ctx.Nx}")
    print(f"Ny = {ctx.Ny}")
    print(f"Nz = {ctx.Nz}")


if __name__ == "__main__":
    main()
