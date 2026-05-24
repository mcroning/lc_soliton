"""
Add a request dataclass name to lc_soliton package-root exports.

Usage:
    python scripts/add_request_export.py GeometryRequest
"""

from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/add_request_export.py NameRequest")

    name = sys.argv[1]
    p = Path("src/lc_soliton/__init__.py")
    s = p.read_text()

    export_line = f'    "{name}",'
    if export_line not in s:
        s = s.replace('    "SimulationRequest",', f'    "SimulationRequest",\n{export_line}')

    loader_line = f'        "{name}",'
    if loader_line not in s:
        s = s.replace('        "SimulationRequest",', f'        "SimulationRequest",\n{loader_line}')

    p.write_text(s)
    print(f"ensured export: {name}")


if __name__ == "__main__":
    main()
