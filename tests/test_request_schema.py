def test_simulation_request_schema():
    from lc_soliton.request_schema import simulation_request_schema

    schema = simulation_request_schema()

    assert "grid" in schema
    assert "solver" in schema
    assert "output" in schema
    assert "runtime" in schema
