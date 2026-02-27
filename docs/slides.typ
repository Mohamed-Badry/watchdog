#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#let color-primary = rgb("#092e4b") // Dark Navy
#let color-secondary = rgb("#e64848") // Soft Red
#let color-code-bg = rgb("#f4f4f4")

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [Project Watchdog],
    subtitle: [AI-Powered Amateur Satellite Ground Station],
    author: [],
  ),
  config-colors(
    primary: color-primary,
    secondary: color-secondary,
  )
)

#set text(font: "New Computer Modern", size: 20pt)

// --- Custom Components ---

#let code-block(content) = {
  block(
    fill: color-code-bg,
    inset: 1em,
    radius: 4pt,
    width: 100%,
    stroke: none, 
    text(size: 0.8em, font: "FiraCode Nerd Font Mono", content)
  )
}

#let task-card(id, title, context_str, instructions, deliverable) = {
  block(
    width: 100%,
    stroke: (left: 4pt + color-secondary),
    fill: rgb("#ffffff"), // Clean white bg
    outset: 0pt,
    inset: (left: 1em, top: 0.5em, bottom: 0.5em),
    radius: 0pt,
    spacing: 1em,
    breakable: false,
  )[
    #text(weight: "bold", fill: color-secondary, size: 1.1em)[#title]
    #v(0.5em)
    #grid(
      columns: (auto, 1fr),
      gutter: 0.8em,
      strong("Goal:"), text(fill: luma(40%))[#context_str],
      strong("Logic:"), text(fill: luma(40%))[#instructions]
    )
    #v(0.6em)
    #block(
      fill: color-primary.lighten(90%), 
      inset: 0.8em, 
      radius: 4pt, 
      width: 100%
    )[
      #text(fill: color-primary, weight: "bold", size: 0.9em)[Result:] #text(size: 0.9em)[#deliverable]
    ]
  ]
}

// --- Slides ---

#title-slide()

== Mission Objective

*Real-Time Anomaly Detection at the Edge*

Instead of passive data logging, this system acts as a "Proactive Early Warning System" for amateur satellites.

#v(1em)
*Key Capabilities:*
- *Universal Decoding:* Leveraging `satnogs-decoders` (Kaitai Structs).
- *Schema Normalization:* Mapping heterogeneous raw telemetry to SI units.
- *Synthetic Fault Injection:* Validating detection using simulated physical failures.

== The Problem vs. The Solution

#text(size: 16pt)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    block(
      fill: color-secondary.lighten(90%),
      // height: 21em,
      inset: 1em,
      radius: 4pt,
      width: 100%,
      stroke: (left: 4pt + color-secondary),
      [
        *The Problem: "Post-Mortem Analysis"*
        #v(0.5em)
        Current amateur ground stations are passive data loggers.
        #v(0.5em)
        - *Latency Gap:* Anomalies are detected weeks later by humans manually reviewing logs.
        #v(0.1em)
        - *Static Thresholds:* Simple limits fail to catch complex, multivariate failures (e.g., thermal runaway during eclipse).
        #v(0.1em)
        - *Scale:* No operator can monitor 300+ live downlinks 24/7.
      ]
    ),
    block(
      fill: color-primary.lighten(90%),
      // height: 21em,
      inset: 1em,
      radius: 4pt,
      width: 100%,
      stroke: (left: 4pt + color-primary),
      [
        *Our Solution: The Edge Watchdog*
        #v(0.5em)
        Deploying an *Unsupervised ML Model* at the Ground Station.
        #v(0.5em)
        - *Zero Latency:* Inference happens *during* the reception window.
        #v(0.1em)
        - *Context Aware:* Learns correlations (e.g., "High Current" is normal *only* if "TX is Active").
        #v(0.1em)
        - *Data Efficient:* Trains on "Normal" data (abundant), doesn't require rare "Failure" labels.
      ]
    )
  )
]

= Phase 1: The Selection Funnel

== The Selection Criteria

To build a reliable detector, we need reliable data. We filtered the entire amateur fleet (338+ satellites).

#v(1em)
#list(
  [*Status:* Must be confirmed 'Alive' in SatNOGS DB.],
  [*Band:* 433-438 MHz (70cm Amateur Band).],
  [*Modulation:* High-rate 9600 bps GMSK/GFSK (Modern Standard).],
  [*Visibility:* High-elevation passes (>30°) over Beni Suef.],
  [*Support:* Must be explicitly supported by `satnogs-decoders` (Kaitai).]
)

== Analysis Result: The Golden Cohort

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    We identified *12 candidates* that match the technical criteria and have decoders.
    
    *Dominant Standard:* 9600 bps GFSK.
    
    This ensures our receiver pipeline works for maximum targets with `satnogs-decoders`.
  ],
  align(center + horizon)[
    #image("figures/modulation_distribution.png", width: 100%)
  ]
)

= Phase 2: Operational Planning

== Heuristic: "Total Observable Time"

Pass count isn't enough. We rank satellites by the *total duration* they spend above 30° elevation over 48 hours.

#task-card("1", "Satellite Scoring", 
  "Maximize data collection opportunity",
  [Sum(Duration of all passes > 30°)],
  "Top Operational Targets"
)

== The Winners (Top Targets)

These satellites offer the highest *Operational Efficiency* for our ground station.

#table(
  columns: (1fr, auto, auto, auto),
  inset: 8pt,
  align: horizon,
  table.header([*Satellite*], [*Passes (48h)*], [*Total Mins*], [*Max El*]),
  "UWE-4", "3", "7.5 m", "50°",
  "BugSat-1 (Dropped)*", "3", "9.5 m", "85°",
)

#v(1em)
*\*Data Engineering Reality Check:* BugSat-1 was dropped despite high visibility due to undocumented protocol variations (`US37` payload header) causing Kaitai parser failures. ML requires clean data; UWE-4 provides perfect compatibility and rich thermal/power telemetry.

== Operational Reality: Skyplot

Where to point the antenna for the Top targets.

#align(center)[
  #image("figures/skyplot_top_candidates.png", height: 85%)
]

== Operational Reality: Schedule

When to operate the ground station (Next 48 Hours).

#align(center)[
  #image("figures/timeline_schedule.png", height: 80%)
]

== Phase 3: System Architecture (Update to satnogsdecoders/kaitai structs)

#align(center)[
  #image("figures/systemArch_compressed.png", height: 100%)
]

== The V-Model Strategy

Two distinct environments sharing a single *Shared Core*.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *A. The Lab (Offline)*
    - Establish Baseline from history
    - Source: SatNOGS DB
    - *Action:* Train Autoencoder
  ],
  [
    *B. The Watchdog (Online)*
    - Detect anomalies live
    - Source: Antenna -> SDR
    - *Action:* Real-time Inference
  ]
)

== Standardization: The Golden Features

#text(style: "italic", fill: luma(50%))[A universal interface for heterogeneous satellite hardware.]

#table(
  columns: (auto, auto, 1fr),
  inset: 10pt,
  align: horizon,
  table.header(
    [*Feature*], [*Unit*], [*Description*]
  ),
  `batt_voltage`, "Volts", "Standardized from mV/ADC",
  `batt_current`, "Amps", "Charge/Discharge rate",
  `temp_obc`, "Celsius", "Main computer temp",
  `solar_current`, "Amps", "Panel health & eclipse",
  `temp_pa`, "Celsius", "PA Temp (Radio stuck ON)",
  `signal_rssi`, "dBm", "RSSI (Tumble detection)"
)

= Implementation Logic

== The Shared Core Pipeline

We use the "Shared Core" (`src/gr_sat/telemetry.py`) which acts as the universal adapter between raw bits and our AI model.

#v(0.5em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    *Pipeline Steps:*
    1.  *Ingest:* Read raw hex frames (SatNOGS/SDR).
    2.  *Parse:* Kaitai Structs via `satnogs-decoders`.
    3.  *Normalize:* Convert binary fields to SI Units (V, A, °C).
    4.  *Validate:* Check against physical limits.
  ],
  [
    *Current Status:*
    - Core Logic: *Operational*
    - Decoder Ecosystem: *satnogs-decoders*
    - Coverage: *80+ Satellites*
  ]
)

== Data Verification (UWE-4)

Telemetry extraction from UWE-4 (NORAD 43880). 
Perfect mapping from `satnogs-decoders` to our Golden Features (Volts, Amps, °C).

#align(center)[
  #image("figures/telemetry_43880.png", height: 80%)
]

== The Inspector Tool

We built an interactive debugger (`telemetry_inspector`) to verify decoders against real historical data.

#align(center)[
  #image("figures/decoded_packets.png", height: 85%)
]

== Summary & Next Steps

1.  *Targeting:* We have locked onto *UWE-4* (43880) as our primary "Golden Path" target.
2.  *Pipeline:* The *Data Refinery* is operational, successfully generating over 2,000 clean training frames.
3.  *Validation:* Inspector tools are in place to verify data quality.

#v(1em)
*Next Milestone:* Train the Autoencoder (The Lab Phase 2).
