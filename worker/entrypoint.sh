#!/bin/sh
# If HATCHET_CLIENT_TOKEN is not set, read it from the file the hatchet_token
# one-shot service wrote into the shared volume (see docker-compose.yml).
set -e
TOKEN_FILE="${HATCHET_CLIENT_TOKEN_FILE:-/hatchet-token/token}"
if [ -z "$HATCHET_CLIENT_TOKEN" ] && [ -r "$TOKEN_FILE" ]; then
  HATCHET_CLIENT_TOKEN="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  export HATCHET_CLIENT_TOKEN
fi
exec "$@"
