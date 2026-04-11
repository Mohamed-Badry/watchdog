#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 11pt,
)

#align(center)[
  #text(size: 22pt, weight: "bold")[UWE-4 Anomaly Detection Dataflow] \
  #v(1em)
  #text(size: 12pt, fill: luma(100))[Project Watchdog Core Architecture]
]

= Pipeline Architecture

The current repository implements an offline telemetry processing and benchmarking pipeline that transforms raw database exports into normalized machine learning inputs. A live inference runtime is still planned, not implemented.

#let node(title, body) = align(center)[
  #block(
    fill: rgb("#fafafa"),
    stroke: 1pt + rgb("#e0e0e0"),
    radius: 4pt,
    width: 80%,
    inset: 10pt,
    align(center)[
      #text(weight: "bold", size: 11pt)[#title] \
      #v(1pt)
      #text(size: 10pt, fill: luma(80))[#body]
    ],
  )
]

#let arrow = align(center)[#v(2pt) #text(size: 12pt, fill: luma(150))[↓] #v(2pt)]

#node([Raw Telemetry], [`data/raw/43880/*.jsonl`])
#arrow
#node([Binary Decoder], [Parses raw HEX strings into Kaitai Struct definitions])
#arrow
#node([Unit Conversion], [Maps internal ADC integers to standardized SI units])
#arrow
#node([Processed Features], [`data/processed/43880.csv`])
#arrow
#node([Standard Scaler], [Normalizes feature distributions to $mu=0, sigma=1$])
#arrow
#node([TelemetryVAE Inference], [Pytorch Variational Autoencoder maps correlation losses])
#arrow
#node([Anomaly Detector], [Offline benchmark currently derives the operating threshold inside evaluation; deployable threshold persistence is not implemented yet])

#v(2em)

= Core Features

To maintain compatibility across differing satellite hardware platforms, all proprietary telemetry is adapted into a "Golden Feature" target. The models train strictly on these standard physical properties:

- `batt_voltage` [V]: Total electrical potential of the power bus.
- `batt_current` [A]: Electrical current. Positive values indicate active solar charging; negative values indicate battery discharging during eclipse phases.
- `temp_batt_a` / `temp_batt_b` [°C]: Temperatures of the physical battery packs. Used primarily to monitor for thermal runaways.
- `temp_panel_z` [°C]: External solar panel temperature. This establishes orbit phase context (daylight heating vs. eclipse cooling) linking current draw expectations to thermal states.

= Limitations of Temporal Features

Initial architecture iterations attempted to detect "stuck" or frozen sensors by calculating rolling variance windows across consecutive frames. That logic was removed from the model input path, but the preprocessing script still emits rolling-variance columns for inspection / legacy analysis.

Low Earth Orbit CubeSats frequently utilize basic Analog-to-Digital Converters (ADCs) with coarse resolution steps (e.g., rigid 1°C gradients). During normal thermal plateau operations, these sensors legitimately report identical sequential integer values over several minutes. This results in a mathematical variance of exactly 0.0.

Because normal ADC step quantization is mathematically indistinguishable from a stuck sensor fault, applying variance-based anomaly rules generated a massive influx of false positives, crippling the pipeline's overall AUROC from 0.78 down to 0.40.

The current benchmark pipeline relies on the Variational Autoencoder (VAE). It focuses on multivariate correlation anomalies rather than univariate variance, but its score threshold is still calibrated in evaluation code rather than stored as a training artifact.

#v(1em)
= Synthetic Fault Benchmarking & Limitations

To compare model behavior, the current repository injects synthesized physical faults into a withheld ~20% chronological test partition of the telemetry.

- *Panel Failure:* We selectively override `batt_current` to a negative draw while `temp_panel_z` indicates direct sunlight.
- *Thermal Runaway:* We artificially surge `temp_batt_a` and `temp_batt_b`, breaking their expected equilibrium with the solar panels and power bus.

*Historical benchmark:* Earlier notebook experiments also included a *Sensor Stuck* fault. That scenario is not part of the current shipped benchmark script.

The current benchmark report should be read as comparative offline analysis, not as a deployment-grade live performance certificate: threshold calibration is still evaluation-derived and VAE scoring is still stochastic in evaluation mode.
