def test_geometry_request_defaults():
    from lc_soliton import GeometryRequest

    g = GeometryRequest()

    assert g.xaper_um > 0
    assert g.yaper_um > 0
    assert g.dz_um > 0
