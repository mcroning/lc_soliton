#!/bin/bash
set -e

REPO=${1:-$HOME/lc_soliton_package_migration/lc_soliton_publishable}
PORT=${PORT:-8501}
PYTHON=${PYTHON:-/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python}

cd "$REPO"

echo "========================================="
echo "LC Soliton Streamlit (OnDemand)"
echo "Repo   : $REPO"
echo "Port   : $PORT"
echo "Python : $PYTHON"
echo "Commit : $(git rev-parse --short HEAD)"
echo "========================================="
echo
echo "Use the Open OnDemand-routed URL for port ${PORT}."
echo

exec "$PYTHON" -m streamlit run app/app.py \
  --server.headless true \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false
