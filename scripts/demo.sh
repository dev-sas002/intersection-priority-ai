#!/usr/bin/env bash
# One-command demo: start the stack, wait for it to be healthy, run a real
# detection against a bundled clip, then drive the signal-priority layer.
#
#   ./scripts/demo.sh                     # builds and starts the compose stack
#   SKIP_COMPOSE=1 ./scripts/demo.sh      # against a server you already started
#   BASE=http://host:port ./scripts/demo.sh
set -euo pipefail

BASE="${BASE:-http://localhost:8250}"
CLIP="${CLIP:-test_files/ambulance_video1.mp4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${SKIP_COMPOSE:-}" ]]; then
  echo "==> Starting the stack"
  docker compose up -d --build
fi

echo "==> Waiting for /health"
for _ in $(seq 1 60); do
  if curl -fsS "$BASE/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "$BASE/health" | python3 -m json.tool

echo
echo "==> POST /predict  ($CLIP)"
curl -fsS -F "file=@${CLIP}" -F "approach=north" "$BASE/predict" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); d["frame"]=("<%d base64 chars>" % len(d["frame"])) if d.get("frame") else None; print(json.dumps(d, indent=2))'

echo
echo "==> Two corroborating sightings on the same approach grant a phase"
for _ in 1 2; do
  curl -fsS -H 'Content-Type: application/json' \
    -d '{"approach":"north","detected":true,"confidence":0.95}' \
    "$BASE/priority/sightings" | python3 -m json.tool
done

echo
echo "==> A competing approach is denied inside the minimum hold"
curl -fsS -H 'Content-Type: application/json' \
  -d '{"approach":"east","detected":true,"confidence":0.99}' \
  "$BASE/priority/sightings" >/dev/null
curl -fsS -H 'Content-Type: application/json' \
  -d '{"approach":"east","detected":true,"confidence":0.99}' \
  "$BASE/priority/sightings" | python3 -m json.tool

echo
echo "==> GET /priority/state"
curl -fsS "$BASE/priority/state" | python3 -m json.tool
