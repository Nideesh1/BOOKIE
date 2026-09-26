#!/usr/bin/env sh
# Create .env and worker/.env from their .env.example files, filling every random
# secret with `openssl rand -hex 32`. Never overwrites an existing file.
# Afterwards edit worker/.env and set MONGODB_URI, MODEL_API_KEY, VOYAGE_API_KEY.
set -eu
cd "$(dirname "$0")/.."

rand() { openssl rand -hex 32; }
setkey() { # file key value
  v=$(printf '%s' "$3" | sed 's/[&|]/\\&/g')
  sed -i.bak "s|^$2=.*|$2=$v|" "$1" && rm -f "$1.bak"
}

if [ -e .env ]; then
  echo ".env exists, leaving it alone"
else
  cp .env.example .env
  for k in CLICKHOUSE_PASSWORD LANGFUSE_REDIS_AUTH NEXTAUTH_SECRET SALT ENCRYPTION_KEY; do
    setkey .env "$k" "$(rand)"
  done
  setkey .env LANGFUSE_PUBLIC_KEY "pk-lf-$(rand | cut -c1-32)"
  setkey .env LANGFUSE_SECRET_KEY "sk-lf-$(rand | cut -c1-32)"
  setkey .env LANGFUSE_PASSWORD "$(rand | cut -c1-16)"
  echo "wrote .env"
fi

if [ -e worker/.env ]; then
  echo "worker/.env exists, leaving it alone"
else
  cp worker/.env.example worker/.env
  pk=$(sed -n 's/^LANGFUSE_PUBLIC_KEY=//p' .env)
  sk=$(sed -n 's/^LANGFUSE_SECRET_KEY=//p' .env)
  if [ -n "$pk" ] && [ -n "$sk" ]; then
    setkey worker/.env OTEL_EXPORTER_OTLP_HEADERS "Authorization=Basic $(printf '%s:%s' "$pk" "$sk" | base64 | tr -d '\n')"
  fi
  echo "wrote worker/.env"
fi

echo
echo "Now fill these in worker/.env: MONGODB_URI, MODEL_API_KEY, VOYAGE_API_KEY"
echo "Langfuse login: you@example.com / $(sed -n 's/^LANGFUSE_PASSWORD=//p' .env)"
