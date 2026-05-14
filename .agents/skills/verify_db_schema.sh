#!/usr/bin/env bash
# Skill: Verify current state of a Postgres table
set -e

TABLE_NAME="$1"

if [ -z "$TABLE_NAME" ]; then
    echo '{"error": "Please provide a table name"}'
    exit 1
fi

DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DB:-gr_sat_db}

OUTPUT=$(docker compose exec -T db psql -U "$DB_USER" -d "$DB_NAME" -c "\d ${TABLE_NAME}" 2>&1)
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "{\"error\": \"Could not describe table ${TABLE_NAME}\", \"details\": \"$OUTPUT\"}"
    exit 1
fi

echo "$OUTPUT" | python3 -c 'import json,sys; print(json.dumps({"table": "'"$TABLE_NAME"'", "schema": sys.stdin.read()}))'