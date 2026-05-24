# GUI

The initial GUI prototype is implemented with Streamlit.

Run from the repository root:

    make gui

or:

    streamlit run app/app.py

The GUI currently exercises the public API:

    LCParams
    run_static

It should not call legacy bridge functions directly.

## Cluster note

On managed clusters, Streamlit access may require VPN, SSH tunneling, or an
institution-provided reverse proxy. The app itself should be launched from the
repository root.

## Reference-case validation

The GUI can display the stored strict-static reference case and validate it
through the public package API.

The validation button calls:

    run_reference_case("strict_static_centroid_drift")

rather than directly invoking legacy scripts.

## Dynamic reference-case selection

The GUI now queries available reference cases dynamically using:

    available_reference_cases()

Reference display uses:

    load_reference_case(name)

Reference validation uses:

    run_reference_case(name)

The GUI no longer directly invokes validation scripts or hardcoded
reference-case filesystem paths.
