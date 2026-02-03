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
    author: [Gemini CLI],
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

Instead of passive data logging, this system acts as a "First Responder" for amateur satellites.

#v(1em)
*Key Capabilities:*
- *Strict Validation:* AX.25 Checksums prevent false identification.
- *Schema Normalization:* Mapping heterogeneous raw telemetry to SI units.
- *Synthetic Fault Injection:* Validating detection using simulated physical failures.

= Phase 1: The Selection Funnel

== The Selection Criteria

To build a reliable detector, we need reliable data. We filtered the entire amateur fleet (338+ satellites).

#v(1em)
#list(
  [*Status:* Must be confirmed 'Alive' in SatNOGS DB.],
  [*Band:* 433-438 MHz (70cm Amateur Band).],
  [*Modulation:* High-rate 9600 bps GMSK/FSK (Modern Standard).],
  [*Visibility:* High-elevation passes (>30°) over Beni Suef.]
)

== Analysis Result: The Golden Cohort

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    We identified *87 candidates* that match the technical criteria.
    
    *Dominant Standard:* 9600 bps GMSK.
    
    This ensures our receiver pipeline (`gr_satellites` "9k6" flow) works for maximum targets.
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
  "Top 5 Operational Targets"
)

== The Winners (Top 5)

These satellites offer the highest *Operational Efficiency* for our ground station.

#table(
  columns: (1fr, auto, auto, auto),
  inset: 8pt,
  align: horizon,
  table.header([*Satellite*], [*Passes (48h)*], [*Total Mins*], [*Max El*]),
  "GO-32 (TechSat-1B)", "4", "20.6 m", "89°",
  "HORYU-4", "7", "18.3 m", "89°",
  "STEP CubeLab-II", "4", "16.9 m", "75°",
  "STRaND-1", "4", "14.4 m", "89°",
  "BisonSat", "4", "13.3 m", "73°",
)

== Operational Reality: Skyplot

Where to point the antenna for the Top 5 targets.

#align(center)[
  #image("figures/skyplot_top_candidates.png", height: 85%)
]

== Operational Reality: Schedule

When to operate the ground station (Next 48 Hours).

#align(center)[
  #image("figures/timeline_schedule.png", height: 80%)
]

== Phase 3: System Architecture

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
  `solar_current`, "Amps", "Panel health & eclipse"
)

== Summary & Next Steps

1.  *Targeting:* We have locked onto *GO-32* and *HORYU-4* as primary targets.
2.  *Operations:* We have a precise schedule for high-quality data.
3.  *Architecture:* The V-Model is ready for implementation.

#v(1em)
*Next Milestone:* Download historical telemetry (The Lab) and train the anomaly detector.
