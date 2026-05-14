#!/usr/bin/env bash
# Wrapper to run specific commands inside running Docker containers safely
# Usage: ./container_exec.sh <service> <command>
set -e

SERVICE="$1"
shift
CMD="$@"

ALLOWED_SERVICES=("api" "db" "frontend" "simulator" "mosquitto")

if [[ ! " ${ALLOWED_SERVICES[*]} " =~ " ${SERVICE} " ]]; then
    echo "Error: Unauthorized service '${SERVICE}'" >&2
    exit 1
fi

# Restrict allowed commands to prevent arbitrary execution
# Examples of allowed commands: "pytest", "alembic upgrade head", "bun run test"
if [[ "$CMD" != *"pytest"* && "$CMD" != *"alembic"* && "$CMD" != *"test"* && "$CMD" != *"bun"* ]]; then
    echo "Error: Unauthorized command. Only tests and migrations are allowed." >&2
    exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$DIR" || exit 1

# Execute safely
OUTPUT=$(docker compose exec -T "$SERVICE" $CMD 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Container execution failed in ${SERVICE}." >&2
    echo "$OUTPUT" >&2
    exit 1
fi

echo "Container execution passed in ${SERVICE}." >&2
exit 0