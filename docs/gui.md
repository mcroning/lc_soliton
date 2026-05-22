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
