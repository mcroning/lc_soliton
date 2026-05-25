def test_request_translation_version_exists():
    from lc_soliton.request_translate import REQUEST_TRANSLATION_VERSION

    assert isinstance(REQUEST_TRANSLATION_VERSION, str)
    assert REQUEST_TRANSLATION_VERSION
