from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type, Protocol
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TelemetryFrame:
    """
    The 'Golden Features' - A standardized interface for all satellite telemetry.
    Units are strictly SI (Volts, Amps, Watts, Celsius, Rad/s).
    
    This is the universal data transfer object (DTO) for the entire system.
    """
    # --- Metadata ---
    timestamp: datetime
    norad_id: int
    source: str  # "satnogs_db" or "live_station"
    raw_frame: bytes = field(repr=False) # Keep original payload for debugging
    
    # --- Power System ---
    batt_voltage: Optional[float] = None  # Volts (V)
    batt_current: Optional[float] = None  # Amps (A) (+Charge / -Discharge)
    solar_power: Optional[float] = None   # Watts (W) (Aggregated)
    
    # --- Thermal ---
    temp_obc: Optional[float] = None      # Celsius (°C) (Main Computer)
    temp_pa: Optional[float] = None       # Celsius (°C) (Power Amplifier - usually hottest)
    temp_batt: Optional[float] = None     # Celsius (°C) (Battery pack)
    
    # --- Attitude / RF ---
    signal_rssi: Optional[float] = None   # dBm (Received Signal Strength)
    tumble_rate: Optional[float] = None   # rad/s (Total magnitude or specific axis)
    
    # --- Status ---
    mode: str = "nominal"                 # e.g., "safe", "nominal", "deployment"
    uptime: Optional[int] = None          # Seconds since boot

    def to_dict(self) -> Dict[str, Any]:
        """Returns a flat dictionary suitable for pandas/parquet."""
        data = {
            k: v for k, v in self.__dict__.items() 
            if k != "raw_frame"
        }
        # Hexify raw frame for storage if needed, or skip it. 
        # For Parquet, skipping raw bytes is usually better to save space unless debugging.
        return data

class BaseDecoder(Protocol):
    """Protocol that all satellite-specific decoders must implement."""
    
    def decode(self, payload: bytes) -> Dict[str, Any]:
        """
        Parses raw bytes into a dictionary of physical values.
        Returns empty dict or None if parsing fails (CRC check, etc).
        """
        ...

class DecoderRegistry:
    """
    Central registry mapping NORAD IDs to their specific Decoder classes.
    """
    _registry: Dict[int, Type[BaseDecoder]] = {}

    @classmethod
    def register(cls, norad_id: int):
        """Decorator to register a decoder class for a specific satellite."""
        def wrapper(decoder_cls: Type[BaseDecoder]):
            cls._registry[norad_id] = decoder_cls
            return decoder_cls
        return wrapper

    @classmethod
    def get_decoder(cls, norad_id: int) -> Optional[BaseDecoder]:
        """Returns an instance of the decoder for the given satellite."""
        decoder_cls = cls._registry.get(int(norad_id))
        if decoder_cls:
            return decoder_cls()
        return None

def process_frame(norad_id: int, payload: bytes, source: str, timestamp: datetime) -> Optional[TelemetryFrame]:
    """
    The Universal Adapter.
    
    1. Looks up decoder.
    2. Decodes raw bytes to SI values.
    3. Wraps in standard TelemetryFrame.
    """
    decoder = DecoderRegistry.get_decoder(norad_id)
    
    if not decoder:
        # No decoder registered for this satellite
        return None
        
    try:
        # Decoder returns a dict of specific fields (volts, temp, etc.)
        decoded_values = decoder.decode(payload)
        
        if not decoded_values:
            return None
            
        # Construct the Standard Frame
        # We merge the decoded values into the dataclass.
        # Unknown fields from the decoder are ignored (or could be stored in a 'meta' dict).
        
        # Filter decoded_values to only match TelemetryFrame fields to avoid TypeErrors
        valid_fields = set(TelemetryFrame.__annotations__.keys())
        filtered_values = {k: v for k, v in decoded_values.items() if k in valid_fields}
        
        return TelemetryFrame(
            timestamp=timestamp,
            norad_id=norad_id,
            source=source,
            raw_frame=payload,
            **filtered_values
        )
        
    except Exception as e:
        logger.warning(f"Failed to decode frame for {norad_id}: {e}")
        return None
