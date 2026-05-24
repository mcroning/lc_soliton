# GUI networking notes

When running Streamlit on a cluster node, Streamlit may print:

    Local URL
    Network URL
    External URL

Typical behavior:

- Local URL works only on the same node.
- Network URL may work from a VPN-connected machine.
- External URL may be blocked by institutional firewalls.

For Tufts cluster testing, the Network URL worked from a Mac connected to VPN.

Do not rely on direct public exposure for production deployment. Future public
deployments should use a proper HTTPS reverse proxy, Open OnDemand integration,
or containerized deployment.
