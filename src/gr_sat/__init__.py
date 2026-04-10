"""
gr_sat — The Shared Core for Project Watchdog.

Provides the telemetry standardization layer (TelemetryFrame), the decoder
registry, and satellite-specific decoder implementations.

Key components:
  - telemetry.TelemetryFrame: The universal "Golden Features" DTO (SI units).
  - telemetry.DecoderRegistry: Maps NORAD IDs to decoder classes.
  - telemetry.process_frame: Full pipeline from raw bytes to TelemetryFrame.
  - decoders/: Satellite-specific decoder implementations.
"""

from .telemetry import TelemetryFrame, DecoderRegistry, process_frame

__all__ = ["TelemetryFrame", "DecoderRegistry", "process_frame"]
