from pathlib import Path

from lc_soliton import load_reference_case
from lc_soliton.reference_translate import reference_config_to_request


def test_strict_static_reference_config_to_typed_request(tmp_path):
    reference_data = load_reference_case("strict_static_centroid_drift")

    req = reference_config_to_request(
        reference_data,
        run_dir=tmp_path / "strict_static_replay",
    )

    assert req.mode == "static"

    assert req.grid.Nx == 256
    assert req.grid.Ny == 256
    assert req.grid.Nz == 50

    assert req.geometry.xaper_um == 75
    assert req.geometry.yaper_um == 100
    assert req.geometry.dz_um == 5
    assert req.geometry.wavelength_um == 0.633

    assert abs(req.material.b - 2.487025357142857) < 1e-12
    assert abs(req.material.bi - 214.28571428571414) < 1e-12
    assert req.material.ne == 1.7
    assert req.material.no == 1.5

    assert req.launch.waist_x_um == 3
    assert req.launch.waist_y_um == 3
    assert req.launch.separation_um == 0
    assert req.launch.coherent is False

    assert req.boundary.use_sponge is True
    assert req.boundary.windowedge == 0.1

    assert req.solver.static_max_steps == 2000
    assert req.output.run_dir == str(tmp_path / "strict_static_replay")
    assert req.output.save_slices is True
