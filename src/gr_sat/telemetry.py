"""
The Shared Core — Telemetry standardization and decoder registry.

This module defines the universal interface for all satellite telemetry in the
system. Every decoder MUST produce data that eventually maps to a TelemetryFrame
containing SI-unit "Golden Features".

Pipeline stages:
    1. decode()  — Raw bytes → Dict of all satellite-specific fields (interim)
    2. adapt()   — Interim Dict → TelemetryFrame (Golden Features, SI units)

Both stages are handled by satellite-specific decoder classes registered via
the DecoderRegistry.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Any, Optional, Type
from datetime import datetime
from abc import ABC, abstractmethod

from loguru import logger


@dataclass
class TelemetryFrame:
    """
    The 'Golden Features' — A standardized interface for all satellite telemetry.
    Units are strictly SI (Volts, Amps, Watts, Celsius).

    This is the universal data transfer object (DTO) for the entire system.
    All decoders MUST map their satellite-specific fields to these.
    """

    # --- Metadata ---
    timestamp: datetime
    norad_id: int
    source: str  # "satnogs_db" or "live_station"

    # --- Identifiers (from AX.25 header) ---
    src_callsign: Optional[str] = None
    dest_callsign: Optional[str] = None

    # --- Power System ---
    batt_voltage: Optional[float] = None    # Volts (V) — Combined/averaged
    batt_current: Optional[float] = None    # Amps (A) — Combined (+Charge / -Discharge)
    batt_a_voltage: Optional[float] = None  # Volts (V) — Battery A
    batt_b_voltage: Optional[float] = None  # Volts (V) — Battery B
    batt_a_current: Optional[float] = None  # Amps (A) — Battery A
    batt_b_current: Optional[float] = None  # Amps (A) — Battery B
    power_consumption: Optional[float] = None  # Watts (W)

    # --- Thermal ---
    temp_obc: Optional[float] = None        # Celsius (°C) — Main Computer
    temp_batt_a: Optional[float] = None     # Celsius (°C) — Battery A
    temp_batt_b: Optional[float] = None     # Celsius (°C) — Battery B
    temp_panel_z: Optional[float] = None    # Celsius (°C) — Solar Panel Z (eclipse proxy)

    # --- Status ---
    uptime: Optional[int] = None            # Seconds since boot

    def to_dict(self) -> Dict[str, Any]:
        """Returns a flat dictionary suitable for CSV/pandas, excluding internal fields."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
        }

    @classmethod
    def field_names(cls) -> set:
        """Returns the set of valid field names for this dataclass."""
        return {f.name for f in fields(cls)}


class BaseDecoder(ABC):
    """
    Abstract base class for satellite-specific decoders.

    Each decoder handles a single satellite type and implements the two-stage
    pipeline:
        1. decode() — Binary payload → raw dict (all Kaitai fields)
        2. adapt()  — Raw dict → TelemetryFrame (SI-unit Golden Features)
    """

    @abstractmethod
    def decode(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Stage 1: Parse raw bytes into a dictionary of all decoded fields.

        Uses satnogs-decoders (Kaitai Structs) for parsing. Returns all fields
        exactly as the Kaitai struct produces them, without unit conversion.
        This output is written to data/interim/.

        Returns None if parsing fails (CRC check, missing fields, etc.).
        """
        ...

    @abstractmethod
    def adapt(self, decoded: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Stage 2: Map satellite-specific decoded fields to Golden Features.

        Performs unit conversions (mV→V, mA→A, raw ADC→°C) and field renaming
        to produce a dict compatible with TelemetryFrame.
        This output is written to data/processed/.

        Returns None if required fields are missing.
        """
        ...


class DecoderRegistry:
    """
    Central registry mapping NORAD IDs to their specific Decoder classes.

    Usage:
        @DecoderRegistry.register(43880)
        class UWE4Decoder(BaseDecoder):
            ...
    """

    _registry: Dict[int, Type[BaseDecoder]] = {}

    @classmethod
    def register(cls, norad_id: int):
        """Decorator to register a decoder class for a specific satellite."""
        def wrapper(decoder_cls: Type[BaseDecoder]):
            cls._registry[norad_id] = decoder_cls
            logger.debug(f"Registered decoder for NORAD {norad_id}: {decoder_cls.__name__}")
            return decoder_cls
        return wrapper

    @classmethod
    def get_decoder(cls, norad_id: int) -> Optional[BaseDecoder]:
        """Returns an instance of the decoder for the given satellite."""
        decoder_cls = cls._registry.get(int(norad_id))
        if decoder_cls:
            return decoder_cls()
        return None

    @classmethod
    def list_supported(cls) -> Dict[int, str]:
        """Returns a dict of {norad_id: decoder_class_name} for all registered decoders."""
        return {nid: dcls.__name__ for nid, dcls in cls._registry.items()}


def process_frame(
    norad_id: int,
    payload: bytes,
    source: str,
    timestamp: datetime,
) -> Optional[TelemetryFrame]:
    """
    The Universal Adapter — Full pipeline from raw bytes to TelemetryFrame.

    1. Looks up the registered decoder for this NORAD ID.
    2. Decodes raw bytes to an interim dict (all Kaitai fields).
    3. Adapts the interim dict to SI-unit Golden Features.
    4. Wraps the result in a TelemetryFrame.
    """
    decoder = DecoderRegistry.get_decoder(norad_id)

    if not decoder:
        logger.warning(f"No decoder registered for NORAD {norad_id}")
        return None

    try:
        # Stage 1: Decode (bytes → raw dict)
        decoded = decoder.decode(payload)
        if not decoded:
            return None

        # Stage 2: Adapt (raw dict → Golden Features dict)
        adapted = decoder.adapt(decoded)
        if not adapted:
            return None

        # Build TelemetryFrame, only passing fields that exist on the dataclass
        valid = TelemetryFrame.field_names()
        filtered = {k: v for k, v in adapted.items() if k in valid}

        return TelemetryFrame(
            timestamp=timestamp,
            norad_id=norad_id,
            source=source,
            **filtered,
        )

    except Exception as e:
        logger.warning(f"Failed to process frame for NORAD {norad_id}: {e}")
        return None
