# Release checklist

## Before tagging a release

### Run tests

    pytest -q tests

### Validate trusted reference case

    lc-soliton --validate-reference

### Check lightweight imports

    python -X importtime -c "import lc_soliton"

### Verify CLI

    lc-soliton --summary
    lc-soliton --env

### Verify Streamlit app launches

    streamlit run app/app.py

### Inspect git status

    git status

Ensure:
- no generated runs
- no __pycache__
- no notebook checkpoints
- no temporary files

### Review documentation

- README
- bootstrap guide
- current status
- portability notes
- public API docs

### Tag release

Example:

    git tag -a v0.0.X -m "Release description"
    git push origin v0.0.X
