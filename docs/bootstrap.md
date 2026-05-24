# Bootstrap / first-time setup

## Clone

    git clone git@github.com:mcroning/lc_soliton.git
    cd lc_soliton

## Install editable package

    pip install -e .

## Verify installation

    lc-soliton --summary
    lc-soliton --validate-reference
    lc-soliton --env

## Run tests

    pytest -q tests

## Launch Streamlit GUI

    streamlit run app/app.py

## Optional run-root override

    export LC_SOLITON_RUN_ROOT=/path/to/runs

## Notes

- Public package imports are intentionally lightweight.
- CuPy/GPU support is optional.
- Reference validation is the primary installation sanity check.
