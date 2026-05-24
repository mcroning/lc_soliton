def test_material_request_defaults():
    from lc_soliton import MaterialRequest

    m = MaterialRequest()

    assert m.ne > m.no
    assert m.K > 0
    assert m.De > 0
