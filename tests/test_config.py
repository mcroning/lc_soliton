from lc_soliton.legacy_validated.lc_core.config import (
    RunConfig,
    config_from_prdata,
    config_to_dict,
    derive_lc_constants,
    validate_config,
)


def test_default_config_derives_and_validates():
    cfg = derive_lc_constants(RunConfig())
    validate_config(cfg)

    assert cfg.grid.Nx == 512
    assert cfg.grid.Ny == 512
    assert cfg.grid.Nz == 200
    assert cfg.grid.xaper_um == cfg.material.d_um
    assert cfg.material.P_W == 1e-3
    assert cfg.material.b > 0
    assert cfg.material.bi > 0


def test_config_from_prdata_preserves_legacy_values():
    prdata = {
        "lm": 0.532,
        "xsamp": 128,
        "ysamp": 256,
        "d": 80.0,
        "yaper": 640.0,
        "rlen": 1000.0,
        "dz": 10.0,
        "P": 2.5,
        "bias_voltage": 3.7,
        "time_behavior": "Time Dependent",
        "tsteps": 50,
        "tend": 0.1,
    }

    cfg = config_from_prdata(prdata)
    validate_config(cfg)

    assert cfg.grid.lm_um == 0.532
    assert cfg.grid.Nx == 128
    assert cfg.grid.Ny == 256
    assert cfg.grid.Nz == 100
    assert cfg.material.d_um == 80.0
    assert cfg.grid.xaper_um == 80.0
    assert cfg.material.P_mW == 2.5
    assert cfg.material.P_W == 2.5e-3
    assert cfg.time.timedep is True
    assert cfg.time.tsteps == 50
    assert abs(cfg.time.dt - 0.002) < 1e-15


def test_config_to_dict_contains_sections():
    cfg = derive_lc_constants(RunConfig())
    d = config_to_dict(cfg)

    assert "grid" in d
    assert "material" in d
    assert "launch" in d
    assert "legacy_prdata" in d
