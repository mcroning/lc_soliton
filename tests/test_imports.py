def test_import_lc_soliton():
    import lc_soliton
    assert lc_soliton.__version__ == "0.0.2"


def test_public_api_star_import():
    namespace = {}
    exec("from lc_soliton import *", namespace)
    assert len(namespace) > 0
