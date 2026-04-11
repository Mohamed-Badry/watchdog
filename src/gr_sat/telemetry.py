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

from dataclasses import dataclass, fields
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


@dataclass(frozen=True)
class ProcessingFailure:
    stage: str
    code: str
    message: str


@dataclass(frozen=True)
class StageOutcome:
    data: Optional[Dict[str, Any]] = None
    failure: Optional[ProcessingFailure] = None

    @property
    def ok(self) -> bool:
        return self.data is not None and self.failure is None


@dataclass(frozen=True)
class FrameProcessingResult:
    frame: Optional[TelemetryFrame] = None
    decoded: Optional[Dict[str, Any]] = None
    adapted: Optional[Dict[str, Any]] = None
    failure: Optional[ProcessingFailure] = None

    @property
    def ok(self) -> bool:
        return self.frame is not None and self.failure is None


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

    def decode_with_diagnostics(self, payload: bytes) -> StageOutcome:
        try:
            decoded = self.decode(payload)
        except Exception as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="decode_exception",
                    message=str(exc),
                )
            )

        if not decoded:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="decode",
                    code="decode_failed",
                    message="Decoder returned no data.",
                )
            )
        return StageOutcome(data=decoded)

    def adapt_with_diagnostics(self, decoded: Dict[str, Any]) -> StageOutcome:
        try:
            adapted = self.adapt(decoded)
        except Exception as exc:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="adapt",
                    code="adapt_exception",
                    message=str(exc),
                )
            )

        if not adapted:
            return StageOutcome(
                failure=ProcessingFailure(
                    stage="adapt",
                    code="adapt_failed",
                    message="Adapter returned no data.",
                )
            )
        return StageOutcome(data=adapted)


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


def process_frame_result(
    norad_id: int,
    payload: bytes,
    source: str,
    timestamp: datetime,
) -> FrameProcessingResult:
    """
    The Universal Adapter — Full pipeline from raw bytes to TelemetryFrame.

    1. Looks up the registered decoder for this NORAD ID.
    2. Decodes raw bytes to an interim dict (all Kaitai fields).
    3. Adapts the interim dict to SI-unit Golden Features.
    4. Wraps the result in a TelemetryFrame.
    """
    decoder = DecoderRegistry.get_decoder(norad_id)

    if not decoder:
        failure = ProcessingFailure(
            stage="process",
            code="no_decoder",
            message=f"No decoder registered for NORAD {norad_id}",
        )
        logger.warning(failure.message)
        return FrameProcessingResult(failure=failure)

    decoded_outcome = decoder.decode_with_diagnostics(payload)
    if not decoded_outcome.ok:
        return FrameProcessingResult(failure=decoded_outcome.failure)

    adapted_outcome = decoder.adapt_with_diagnostics(decoded_outcome.data)
    if not adapted_outcome.ok:
        return FrameProcessingResult(
            decoded=decoded_outcome.data,
            failure=adapted_outcome.failure,
        )

    try:
        # Build TelemetryFrame, only passing fields that exist on the dataclass
        valid = TelemetryFrame.field_names()
        filtered = {k: v for k, v in adapted_outcome.data.items() if k in valid}

        return FrameProcessingResult(
            frame=TelemetryFrame(
                timestamp=timestamp,
                norad_id=norad_id,
                source=source,
                **filtered,
            ),
            decoded=decoded_outcome.data,
            adapted=adapted_outcome.data,
        )

    except Exception as exc:
        failure = ProcessingFailure(
            stage="process",
            code="frame_build_exception",
            message=str(exc),
        )
        logger.warning(f"Failed to process frame for NORAD {norad_id}: {exc}")
        return FrameProcessingResult(
            decoded=decoded_outcome.data,
            adapted=adapted_outcome.data,
            failure=failure,
        )


def process_frame(
    norad_id: int,
    payload: bytes,
    source: str,
    timestamp: datetime,
) -> Optional[TelemetryFrame]:
    result = process_frame_result(norad_id, payload, source, timestamp)
    return result.frame
