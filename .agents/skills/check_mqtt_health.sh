#!/usr/bin/env bash
# Skill: Check MQTT Broker health and simulator connectivity
set -e

# Uses docker compose to ping the mosquitto container or run a quick publish/subscribe check
# We run mosquitto_sub in background, then publish a message, then wait for the sub to receive it.
OUTPUT=$(docker compose exec -T mosquitto sh -c 'mosquitto_sub -t "health/check" -W 2 -C 1 & sleep 0.5; mosquitto_pub -t "health/check" -m "ok"')

if echo "$OUTPUT" | grep -q "ok"; then
    echo '{"status": "healthy", "message": "MQTT Broker is responsive."}'
else
    echo '{"status": "unhealthy", "message": "Failed to verify MQTT broker communication."}'
    exit 1
fi