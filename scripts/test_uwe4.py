import requests
import os
from dotenv import load_dotenv
import satnogsdecoders.decoder.uwe4 as uwe4
import satnogsdecoders.decoder as dec
from kaitaistruct import KaitaiStream, BytesIO

load_dotenv()
token = os.getenv("SATNOGS_API_TOKEN")
headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
url = "https://db.satnogs.org/api/telemetry/?satellite=43880&format=json&page=1"

print("Fetching UWE-4 data...")
resp = requests.get(url, headers=headers)
if resp.status_code == 200:
    data = resp.json()
    frames = data if isinstance(data, list) else data.get("results", [])
    print(f"Got {len(frames)} frames.")
    
    success = 0
    for f in frames:
        hex_str = f.get("frame")
        if not hex_str: continue
        try:
            raw = bytes.fromhex(hex_str)
            struct = uwe4.Uwe4.from_bytes(raw)
            fields = dec.get_fields(struct)
            if fields:
                print(f"Decoded: {fields}")
                success += 1
                break # Just need one good one
        except Exception as e:
            pass
    print(f"Success: {success}")
else:
    print(f"Error fetching: {resp.status_code}")
