"""
Satellite Decoders Package.

This package contains satellite-specific decoder implementations. Each decoder
is a subclass of BaseDecoder and is automatically registered in the
DecoderRegistry via the @DecoderRegistry.register(norad_id) decorator.

To add a new satellite decoder:
  1. Create a new file in this directory (e.g., inspiresat1.py).
  2. Subclass BaseDecoder and implement decode() and adapt().
  3. Decorate the class with @DecoderRegistry.register(NORAD_ID).
  4. Import it below.

Currently supported satellites:
  - UWE-4 (NORAD 43880) — src/gr_sat/decoders/uwe4.py
"""

from . import uwe4  # noqa: F401