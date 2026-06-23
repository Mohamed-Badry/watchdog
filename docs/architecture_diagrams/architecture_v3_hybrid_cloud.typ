#set page(width: auto, height: auto, margin: 1.5cm, fill: rgb("f8fafc"))
#set text(size: 11pt, font: ("Inter", "Arial", "sans-serif"))

#let icon(name, w: 1.2cm) = image("icons/" + name + ".svg", width: w)

#let component(ic, title, desc, bg: white, border: luma(200)) = rect(
  fill: bg, stroke: 1.5pt + border, radius: 8pt, width: 100%, inset: 12pt,
  [
    #grid(
      columns: (1.5cm, 1fr),
      align: horizon + left,
      column-gutter: 15pt,
      icon(ic, w: 1.4cm),
      [
        *#text(size: 12pt)[#title]* \
        #text(size: 9pt, fill: luma(100))[#desc]
      ]
    )
  ]
)

#let down-pipe(label) = align(center)[
  #v(-2pt)
  #rect(fill: white, stroke: 1pt + rgb("94a3b8"), radius: 4pt, inset: 6pt)[
    #text(size: 9pt, weight: "bold", fill: rgb("475569"))[#label]
  ]
  #v(-8pt)
  #text(size: 28pt, fill: rgb("94a3b8"))[↓]
  #v(-8pt)
]

#let right-pipe(label) = align(center)[
  #rect(fill: white, stroke: 1.5pt + rgb("ef4444"), radius: 6pt, inset: 10pt)[
    #text(size: 11pt, weight: "bold", fill: rgb("b91c1c"))[#label]
  ] \
  #text(size: 40pt, fill: rgb("ef4444"))[→] \
]

#align(center)[
  #text(size: 24pt, weight: "black", fill: rgb("0f172a"))[Hybrid Edge-to-Cloud Deployment]
  #v(2.5em)
]

#grid(
  columns: (9cm, 4cm, 14cm, 4cm, 7cm),
  align: center + horizon,
  
  // EDGE / GROUND STATION
  rect(
    width: 100%, fill: rgb("ecfdf5"), stroke: 2pt + rgb("10b981"), radius: 12pt, inset: 20pt,
    [
      #align(left)[
        #text(size: 16pt, weight: "bold", fill: rgb("065f46"))[Ground Station] \
        #text(size: 11pt, fill: rgb("047857"))[_Local Laptop @ Antenna_]
      ]
      #v(20pt)
      #component("antenna", "Live Antenna & SDR", "Captures raw RF", border: rgb("10b981"))
      #down-pipe("Raw IQ Samples (UDP)")
      #component("python", "Decoder Pipeline", "Demodulates & parses Hex frames", border: rgb("10b981"))
      #down-pipe("Decoded JSON Packets")
      #component("mqtt", "Telemetry Relay", "Publishes JSON frames to Cloud", border: rgb("10b981"))
      
      #v(25pt)
      #rect(fill: rgb("fef3c7"), stroke: 1.5pt + rgb("f59e0b"), radius: 6pt, width: 100%, inset: 12pt)[
        #text(size: 10pt, weight: "bold", fill: rgb("b45309"))[Offline Fallback Buffer] \
        #text(size: 9pt, fill: rgb("b45309"))[Local CSV backup in case Wi-Fi drops]
      ]
    ]
  ),
  
  // NETWORK 1
  right-pipe("Encrypted MQTT\n(Internet)"),
  
  // SATNOGS + CLOUD VPS
  [
    #rect(width: 100%, fill: rgb("eff6ff"), stroke: 2pt + rgb("3b82f6"), radius: 12pt, inset: 15pt)[
      #grid(
        columns: (1.5cm, 1fr),
        align: horizon + left,
        column-gutter: 15pt,
        icon("satnogs", w: 1.4cm),
        [
          #text(size: 14pt, weight: "bold", fill: rgb("1d4ed8"))[SatNOGS Global Network] \
          #text(size: 10pt, fill: rgb("1e40af"))[Global Database of Satellite Telemetry]
        ]
      )
    ]
    #down-pipe("REST API Polling")
    #rect(
      width: 100%, fill: rgb("f5f3ff"), stroke: 2pt + rgb("8b5cf6"), radius: 12pt, inset: 20pt,
      [
        #grid(
          columns: (auto, 1fr), align: horizon + left, column-gutter: 15pt,
          icon("docker", w: 1.4cm),
          [
            #text(size: 16pt, weight: "bold", fill: rgb("4c1d95"))[Cloud VPS] \
            #text(size: 11pt, fill: rgb("5b21b6"))[_Docker Compose Stack_]
          ]
        )
        #v(20pt)
        #component("mqtt", "MQTT Broker (Mosquitto)", "Receives & authenticates streams", border: rgb("8b5cf6"))
        #down-pipe("MQTT Sub (paho-mqtt)")
        #component("fastapi", "AI Backend (FastAPI)", "Normalizes SI Units", border: rgb("8b5cf6"))
        #down-pipe("1D Tensors (Memory)")
        #component("pytorch", "Inference Engine", "PyTorch VAE Anomaly Scoring", border: rgb("8b5cf6"))
        #v(20pt)
        
        #grid(
          columns: (1fr, 1fr),
          align: center,
          column-gutter: 20pt,
          [
            #text(size: 24pt, fill: rgb("94a3b8"))[↙] \
            #text(size: 10pt, weight: "bold", fill: rgb("64748b"))[_psycopg2 (SQL)_] \
            #v(5pt)
            #component("postgresql", "TimescaleDB", "Persistence", bg: white, border: rgb("8b5cf6"))
          ],
          [
            #text(size: 24pt, fill: rgb("94a3b8"))[↘] \
            #text(size: 10pt, weight: "bold", fill: rgb("64748b"))[_WebSockets (JSON)_] \
            #v(5pt)
            #component("svelte", "SvelteKit UI", "Dashboard", bg: white, border: rgb("8b5cf6"))
          ]
        )
        
        #v(25pt)
        #rect(fill: rgb("e0e7ff"), stroke: 1.5pt + rgb("4f46e5"), radius: 6pt, width: 100%, inset: 12pt)[
          #text(size: 10pt, weight: "bold", fill: rgb("3730a3"))[watchdog_scheduler (Cron)] \
          #text(size: 9pt, fill: rgb("3730a3"))[Daily fetch & retrain + Webhook cache invalidation]
        ]
      ]
    )
  ],
  
  // NETWORK 2
  right-pipe("HTTPS / WSS\n(WebSockets)"),
  
  // USERS
  rect(
    width: 100%, fill: white, stroke: 2pt + rgb("64748b"), radius: 12pt, inset: 20pt,
    [
      #align(left)[
        #text(size: 16pt, weight: "bold", fill: rgb("334155"))[End Users]
      ]
      #v(20pt)
      #rect(width: 100%, fill: rgb("f1f5f9"), stroke: none, radius: 8pt, inset: 20pt)[
        #align(center)[
          #icon("svelte", w: 2.5cm) \
          #v(15pt)
          #text(size: 14pt, weight: "bold")[Public Web Browser] \
          #v(5pt)
          #text(size: 11pt, fill: luma(100))[Views live satellite health]
        ]
      ]
    ]
  )
)
