#set page(width: 800pt, height: auto, margin: 2cm, fill: rgb("f8fafc"))
#set text(font: ("Arial", "sans-serif"), size: 10pt)

#let icon(name, w: 1.2cm) = image("icons/" + name + ".svg", width: w)

#let container(name, title, desc, border_color) = rect(
  fill: white, stroke: 1.5pt + border_color, radius: 8pt, width: 240pt, inset: 12pt,
  grid(
    columns: (1.5cm, 1fr),
    align: horizon + left,
    icon(name, w: 1.3cm),
    [
      *#text(size: 11pt, fill: border_color)[#title]* \
      #text(size: 9pt, fill: luma(100))[#desc]
    ]
  )
)

#let v-arrow(label) = align(center)[
  #v(8pt)
  #rect(fill: white, stroke: 1pt + rgb("cbd5e1"), radius: 4pt, inset: 6pt)[
    #text(size: 9pt, weight: "bold", fill: rgb("64748b"))[#label]
  ]
  #v(-5pt)
  #text(size: 24pt, fill: rgb("cbd5e1"))[↓]
  #v(8pt)
]

#align(center)[
  #text(size: 24pt, weight: "bold", fill: rgb("0f172a"))[Watchdog Docker Architecture]
  #v(1em)
  #text(size: 12pt, fill: rgb("64748b"))[Container topology and volume mounts]
  #v(2em)

  #rect(fill: rgb("f1f5f9"), stroke: 2pt + rgb("94a3b8"), radius: 12pt, inset: 30pt, width: 680pt)[
    #align(left)[
      #text(size: 16pt, weight: "bold", fill: rgb("334155"))[Docker Host (vps-watchdog)]
    ]
    #v(20pt)

    // TIER 1: Presentation
    #rect(fill: rgb("fff1f2"), stroke: 1.5pt + rgb("f43f5e"), radius: 8pt, inset: 20pt, width: 620pt)[
      #align(left)[#text(size: 12pt, weight: "bold", fill: rgb("e11d48"))[Presentation Tier]]
      #v(15pt)
      #align(center)[
        #container("svelte", "watchdog_frontend", "SvelteKit UI (Port 5173)", rgb("e11d48"))
      ]
    ]

    #v-arrow("REST API / WebSockets")

    // TIER 2: Application
    #rect(fill: rgb("eff6ff"), stroke: 1.5pt + rgb("3b82f6"), radius: 8pt, inset: 20pt, width: 620pt)[
      #align(left)[#text(size: 12pt, weight: "bold", fill: rgb("2563eb"))[Application Tier]]
      #v(15pt)
      #align(center)[
        #container("fastapi", "watchdog_api", "FastAPI Core (Port 8000)", rgb("3b82f6"))
      ]
    ]

    #v-arrow("Internal Routing")

    // TIER 3: Backend Services
    #rect(fill: rgb("f8fafc"), stroke: 1.5pt + rgb("cbd5e1"), radius: 8pt, inset: 20pt, width: 620pt)[
      #align(left)[#text(size: 12pt, weight: "bold", fill: rgb("475569"))[Background & Data Services]]
      #v(15pt)
      
      #align(center)[
        #grid(
          columns: (240pt, 240pt),
          align: center + horizon,
          column-gutter: 40pt,
          row-gutter: 20pt,
          
          container("mqtt", "watchdog_broker", "Mosquitto (Port 1883)", rgb("8b5cf6")),
          container("postgresql", "watchdog_db", "TimescaleDB (Port 5432)", rgb("0ea5e9")),
          
          container("python", "watchdog_simulator", "Mock Telemetry Generator", rgb("10b981")),
          container("python", "watchdog_scheduler", "Cron: Fetch & Train", rgb("f59e0b"))
        )
      ]
    ]
    
    #v(40pt)
    
    // Shared Volumes Footer
    #rect(fill: white, stroke: 1.5pt + rgb("94a3b8"), radius: 8pt, inset: 20pt, width: 620pt)[
      #align(center)[
        #text(size: 12pt, weight: "bold", fill: rgb("334155"))[Shared Docker Volumes]
        #v(15pt)
        #grid(
          columns: (auto, auto, auto, auto),
          column-gutter: 40pt,
          align: center,
          [ *`./src`* \ #text(size: 10pt, fill: rgb("64748b"))[Live code mounts] ],
          [ *`./data`* \ #text(size: 10pt, fill: rgb("64748b"))[Raw & processed CSVs] ],
          [ *`./models`* \ #text(size: 10pt, fill: rgb("64748b"))[PyTorch `.pt` artifacts] ],
          [ *`timescaledb_data`* \ #text(size: 10pt, fill: rgb("64748b"))[PostgreSQL storage] ]
        )
      ]
    ]
  ]
]
