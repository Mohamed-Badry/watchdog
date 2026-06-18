import os
import json
import time
import logging
import random
from datetime import datetime, timezone
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Simulator")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Simulator connected to MQTT broker")
        client.connected_flag = True
    else:
        logger.error(f"Simulator failed to connect: {rc}")
        client.connected_flag = False

def on_disconnect(client, userdata, rc):
    logger.warning("Simulator disconnected from MQTT broker")
    client.connected_flag = False

def main():
    broker_url = os.getenv("MQTT_BROKER_URL", "localhost")
    broker_port = int(os.getenv("MQTT_BROKER_PORT", 1883))
    username = os.getenv("MQTT_USERNAME")
    password = os.getenv("MQTT_PASSWORD")
    use_tls = os.getenv("MQTT_USE_TLS", "false").lower() == "true"

    client = mqtt.Client()
    client.connected_flag = False
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    
    if username and password:
        client.username_pw_set(username, password)
    
    if use_tls:
        client.tls_set()

    try:
        client.connect(broker_url, broker_port, 60)
        client.loop_start()
    except Exception as e:
        logger.error(f"Simulator initial connection failed: {e}")
        # We don't return, we let the loop try to publish and use the fallback buffer


    # Simulate reading from data/raw/
    raw_dir = "/app/data/raw" if os.path.exists("/app/data/raw") else "../../data/raw"
    if not os.path.exists(raw_dir):
        logger.warning(f"Raw data dir {raw_dir} not found. Mocking random data.")
        mock_mode = True
    else:
        mock_mode = False
        # Read a sample JSONL file
        files = [f for f in os.listdir(raw_dir) if f.endswith(".jsonl")]
        if files:
            with open(os.path.join(raw_dir, files[0])) as f:
                lines = f.readlines()
        else:
            mock_mode = True

    idx = 0
    while True:
        if mock_mode:
            norad_id = 43880
            # A valid UWE-4 raw packet header mock
            raw_frame = "8A8A8A8A8A8A608A8A8A8A8A8A6103F0" + "".join(
                [random.choice("0123456789ABCDEF") for _ in range(64)]
            )
        else:
            try:
                record = json.loads(lines[idx % len(lines)])
                norad_id = record.get("norad_id", 43880)
                raw_frame = record.get("frame", "")
                idx += 1
            except:
                norad_id = 43880
                raw_frame = "8A8A8A8A8A8A"

        payload = {
            "norad_id": norad_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_frame": raw_frame,
            "station_id": "sim_station_1",
            "snr": round(random.uniform(5.0, 25.0), 1),
        }

        payload_str = json.dumps(payload)
        
        if getattr(client, "connected_flag", False):
            info = client.publish(f"telemetry/live/{norad_id}", payload_str, qos=1)
            if info.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published to telemetry/live/{norad_id}")
            else:
                logger.error(f"Publish failed (rc={info.rc}), using offline fallback")
                _write_fallback(payload)
        else:
            logger.warning("MQTT disconnected. Writing to offline fallback buffer.")
            _write_fallback(payload)
            
        time.sleep(5)

def _write_fallback(payload):
    import csv
    fallback_path = "/app/data/raw/fallback_buffer.csv" if os.path.exists("/app/data/raw") else "../../data/raw/fallback_buffer.csv"
    os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
    
    file_exists = os.path.isfile(fallback_path)
    try:
        with open(fallback_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'norad_id', 'station_id', 'snr', 'raw_frame']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(payload)
    except Exception as e:
        logger.error(f"Failed to write to fallback buffer: {e}")


if __name__ == "__main__":
    time.sleep(10)  # Wait for broker to start
    main()
