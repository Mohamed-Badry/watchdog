#import "@preview/touying:0.5.3": *
#import themes.metropolis: *

#let color-primary = rgb("#092e4b") // Dark Navy
#let color-secondary = rgb("#e64848") // Soft Red
#let color-code-bg = rgb("#f4f4f4")

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [UWE-4 Exploratory Data Analysis],
    subtitle: [Telemetry Validation, Benchmarking & Dashboard Readiness],
    author: [],
  ),
  config-colors(
    primary: color-primary,
    secondary: color-secondary,
  ),
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
    text(size: 0.8em, font: "FiraCode Nerd Font Mono", content),
  )
}

#let task-card(id, title, context_str, instructions, deliverable) = {
  block(
    width: 100%,
    stroke: (left: 4pt + color-secondary),
    fill: rgb("#ffffff"),
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
      strong("Context:"), text(fill: luma(40%))[#context_str],
      strong("Action:"), text(fill: luma(40%))[#instructions],
    )
    #v(0.6em)
    #block(
      fill: color-primary.lighten(90%),
      inset: 0.8em,
      radius: 4pt,
      width: 100%,
    )[
      #text(fill: color-primary, weight: "bold", size: 0.9em)[Key Insight:] #text(size: 0.9em)[#deliverable]
    ]
  ]
}

// --- Slides ---

#title-slide()

= 1. Data Engineering & Sanity Checks

== Pipeline Integrity & Feature Distributions

Our core dataset spans 7 months of clean SI-unit data. We actively filter out known physical impossibilities early (e.g. 8V battery spikes) because they are communications errors, not physics anomalies.

#align(center)[
  #image("figures/feature_distributions.png", width: 85%)
]

= 2. Deep-Dive Exploratory Data Analysis

== Long-Term Macro Trends (7 Months)

#text(
  size: 16pt,
)[Over the 7-month dataset, we see extreme seasonality in satellite thermodynamics and charge cycles. The baselines shift massively over the year.]

#align(center)[
  #image("figures/timeseries_macro_7month.png", width: 95%)
]

== The Bimodality Challenge (Day vs. Eclipse)

#grid(
  columns: (1fr, 1.2fr),
  gutter: 1.5em,
  align(center)[
    #image("figures/eclipse_scatter.png", width: 100%)
  ],
  [
    #v(1em)
    *Non-Linear Correlations & States*

    The transition into eclipse structurally uncouples features. `batt_current` acts differently when `temp_panel_z < 0` (Eclipse) versus when it's `> 20` (Sunlight).

    A static threshold logic gate would either trigger 1,000 false positives every orbit, or be so loose that it misses catastrophic component failures.
  ],
)

== Principal Component Analysis (Latent Structure)

#text(
  size: 16pt,
)[PCA proves our 5 physical features are heavily correlated and can be mathematically compressed, validating our Autoencoder hypothesis.]

#align(center)[
  #image("figures/pca_analysis.png", width: 90%)
]

= 3. The Edge Discontinuity

== Time Gaps vs Traditional Time-Series

Edge station inference is strictly constrained by Line-Of-Sight passes.

#grid(
  columns: (1fr, 1fr),
  gutter: 1.5em,
  [
    #v(1em)
    - Median gap *within* a pass: ~10-15s
    - Median gap *between* passes: ~10 hours!

    *Why LSTMs / Transformers fail here:*
    Rolling history expires between passes. Attempting to use the frame from "10 hours ago" breaks physics predictions.

    *Conclusion:* We require *Stateless Ensembles* (evaluating each frame in a vacuum).
  ],
  align(center)[
    #image("figures/time_gap_distribution.png", width: 100%)
  ],
)

= 4. Model Benchmarking

== The Evaluated AI Models

We benchmarked 4 models against 3 synthetically injected physical faults (`Sensor Stuck`, `Thermal Runaway`, and `Panel Failure`).

#align(center)[
  #image("figures/model_comparison_roc.png", width: 65%)
]


#pagebreak()

*The Standouts:*
#v(0.5em)
- *Elliptic Envelope:* Appeared to do well in basic AUROC, but suffered catastrophic ~50% Recall failure because its Gaussian boundaries cannot map bimodal clusters (Day/Night) accurately without drawing boundaries too loose.
- *Variational Autoencoder (VAE):* Perfect baseline handling non-linear space with explicit probabilistic modeling.

== The Unified Architecture (PyTorch VAE)

#task-card(
  "1",
  "Stage 1: Overall Loss Detector",
  "We dumped the EllipticEnvelope entirely.",
  "Run the VAE. Calculate the sum of its Mean Squared Error (MSE) + Kullback-Leibler Divergence (KLD). If it exceeds the 95th Percentile training threshold, flag anomalous.",
  "Uses the VAE's intrinsic ability to map non-linear bimodal physics mathematically.",
)

#v(1em)

#task-card(
  "2",
  "Stage 2: Per-Node Diagnoser",
  "We must isolate what broke during an anomaly.",
  "Inspect the Mean Squared Error node-by-node. The feature with the highest individual MSE represents the hardware failure.",
  "Successfully spots 100% of Thermal Runaways on edge benchmarks!",
)


= 5. Dashboard Blueprints

== Micro Scale Pass Dynamics (Widget #1)

#text(size: 16pt)[A live time-series tracking variables throughout a specific satellite pass overhead.]

#align(center)[
  #image("figures/pass_dynamics_micro.png", width: 90%)
]

== Autoencoder Feature Imputation (Widget #2)

#text(
  size: 16pt,
)[When Stage 2 triggers, the UI will display *Actual Values* against the Autoencoder's *Reconstructed/Expected Values* to show the exact delta forcing the anomaly.]

#align(center)[
  #image("figures/ae_feature_contribution.png", width: 90%)
]

== Final Live Dashboard Strategy

To productize "The Watchdog", we will transition these findings into a unified `marimo` Reactive Notebook for the operator UI.

*Key Dashboard Widgets to build:*
1. *Live Pass View:* Telemetry graphing in real-time as packets decode.
2. *Gauge Cluster:* Real-time dials mapping values to historical +/- 3 Sigma bounds.
3. *Anomaly Probability Dial:* Stage 1 `EllipticEnvelope` score mapped from 0-100%.
4. *Subsystem Blame Chart:* Stage 2 `Autoencoder` feature-reconstruction radar chart.
