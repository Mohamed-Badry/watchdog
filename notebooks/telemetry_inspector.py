# %% [markdown]
# # Telemetry Inspector
# **Interactive Debugger for the "Shared Core" Pipeline**
#
# This notebook allows you to trace the lifecycle of a single packet:
# `Raw Hex` -> `Binary Struct` -> `Standardized Telemetry`
#
# Use this to verify decoders against real SatNOGS data.

# %%
import ipywidgets as widgets
from IPython.display import display
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import textwrap
import html
import sys

# Project Imports
sys.path.append(str(Path("..").resolve() / "src"))
sys.path.append(str(Path(".").resolve() / "src"))

from gr_sat.telemetry import process_frame
from gr_sat.decoders.go32 import GO32_Struct

# Config
RAW_DIR = Path("../data/raw")
if not RAW_DIR.exists():
    RAW_DIR = Path("data/raw")

# %% [markdown]
# ## 1. Helper Functions

# %%
def load_frames(norad_id, filename):
    filepath = RAW_DIR / str(norad_id) / filename
    frames = []
    if not filepath.exists():
        return []
    with open(filepath, 'r') as fh:
        for line in fh:
            try:
                frames.append(json.loads(line))
            except:
                continue
    return frames

def dict_to_html(data, title="Data"):
    """
    Recursively converts a dictionary to a nice HTML tree/list.
    """
    if not data:
        return f"<div style='color: #888; font-style: italic;'>Empty</div>"
        
    html_out = f"<div style='font-family: monospace;'>"
    
    if "error" in data:
        return f"<div style='color: red; font-weight: bold;'>Error: {html.escape(str(data['error']))}</div>"
        
    for k, v in data.items():
        key_style = "color: #0066cc; font-weight: bold;"
        val_style = "color: #333;"
        
        if isinstance(v, dict):
            # Nested
            html_out += f"<div style='margin-bottom: 4px;'>"
            html_out += f"<span style='{key_style}'>{k}</span>:"
            html_out += f"<div style='margin-left: 15px; border-left: 2px solid #eee; padding-left: 5px;'>"
            html_out += dict_to_html(v, "")
            html_out += "</div></div>"
        else:
            # Leaf
            if isinstance(v, float):
                v_str = f"{v:.4f}"
                val_style = "color: #b35900;" # Orange-ish for numbers
            elif isinstance(v, int):
                v_str = str(v)
                val_style = "color: #b35900;"
            else:
                v_str = html.escape(str(v))
                val_style = "color: #228b22;" # Greenish for strings
            
            # Unit guessing for display
            unit = ""
            k_lower = k.lower()
            if "volt" in k_lower: unit = " V"
            elif "curr" in k_lower: unit = " A"
            elif "temp" in k_lower: unit = " °C"
            elif "rssi" in k_lower: unit = " dBm"
            elif "power" in k_lower: unit = " W"
            
            html_out += f"<div><span style='{key_style}'>{k}</span>: <span style='{val_style}'>{v_str}</span><span style='color: #999; font-size: 0.8em;'>{unit}</span></div>"
            
    html_out += "</div>"
    return html_out

# %% [markdown]
# ## 2. Inspector UI

# %%
class TelemetryInspector:
    def __init__(self):
        self.sats = self._get_available_satellites()
        
        # State
        self.frames = []
        self.current_norad = self.sats[0] if self.sats else None
        
        # --- UI Components ---
        
        # 1. Controls
        self.sat_dropdown = widgets.Dropdown(
            options=self.sats,
            value=self.current_norad,
            description='Satellite:',
            layout=widgets.Layout(width='250px')
        )
        self.day_dropdown = widgets.Dropdown(
            options=[],
            description='Day:',
            layout=widgets.Layout(width='200px')
        )
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=0, step=1,
            description='Frame:',
            continuous_update=False,
            layout=widgets.Layout(width='100%')
        )
        self.btn_prev = widgets.Button(description="<", layout=widgets.Layout(width='40px'))
        self.btn_next = widgets.Button(description=">", layout=widgets.Layout(width='40px'))
        
        # 2. Panels (HTML Widgets)
        common_layout = widgets.Layout(
            width='33%', 
            border='1px solid #ddd', 
            height='400px', 
            overflow='auto',
            padding='10px',
            margin='5px'
        )
        
        self.panel_raw = widgets.HTML(layout=common_layout)
        self.panel_struct = widgets.HTML(layout=common_layout)
        self.panel_telem = widgets.HTML(layout=common_layout)
        
        # 3. Header
        self.header_label = widgets.HTML("<h3>No Data Loaded</h3>")

        # --- Bindings ---
        self.sat_dropdown.observe(self._on_sat_change, names='value')
        self.day_dropdown.observe(self._on_day_change, names='value')
        self.frame_slider.observe(self._on_frame_change, names='value')
        self.btn_prev.on_click(self._on_prev)
        self.btn_next.on_click(self._on_next)
        
        # --- Initialization ---
        if self.current_norad:
            self._load_days(self.current_norad)

    def _get_available_satellites(self):
        if not RAW_DIR.exists(): return []
        return sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])

    def _get_available_days(self, norad_id):
        sat_dir = RAW_DIR / str(norad_id)
        if not sat_dir.exists(): return []
        return sorted([f.name for f in sat_dir.glob("*.jsonl")])

    def _load_days(self, norad_id):
        days = self._get_available_days(norad_id)
        
        try: self.day_dropdown.unobserve(self._on_day_change, names='value')
        except ValueError: pass
        
        self.day_dropdown.options = days
        
        if days:
            self.day_dropdown.value = days[0]
            self._load_frames(norad_id, days[0])
        else:
            self.day_dropdown.value = None
            self.frames = []
            self._update_slider()
            self.render()
            
        self.day_dropdown.observe(self._on_day_change, names='value')

    def _load_frames(self, norad_id, day):
        self.frames = load_frames(norad_id, day)
        self._update_slider()
        self.render()

    def _update_slider(self):
        try: self.frame_slider.unobserve(self._on_frame_change, names='value')
        except ValueError: pass
        
        self.frame_slider.value = 0
        self.frame_slider.max = max(0, len(self.frames) - 1)
        
        self.frame_slider.observe(self._on_frame_change, names='value')

    def render(self):
        if not self.frames:
            self.header_label.value = "<h3>No frames available</h3>"
            self.panel_raw.value = ""
            self.panel_struct.value = ""
            self.panel_telem.value = ""
            return
            
        idx = self.frame_slider.value
        if idx >= len(self.frames): return
        
        frame = self.frames[idx]
        self._render_frame(self.current_norad, idx, frame)

    def _render_frame(self, norad_id, idx, frame):
        raw_hex = frame.get('frame', '')
        timestamp = frame.get('timestamp', '')
        
        # Update Header
        self.header_label.value = f"<h3>Frame #{idx} | {timestamp} | NORAD {norad_id}</h3>"
        
        # 1. Raw Panel
        try:
            raw_bytes = bytes.fromhex(raw_hex)
            # Add spaces every 2 bytes for readability
            hex_str = raw_hex.upper()
            spaced_hex = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
            # Wrap
            wrapped_hex = textwrap.fill(spaced_hex, width=48)
            self.panel_raw.value = f"<h4>Raw Payload ({len(raw_bytes)}B)</h4><pre style='background:#f4f4f4; padding:5px; border-radius:3px;'>{wrapped_hex}</pre>"
        except:
            raw_bytes = b""
            self.panel_raw.value = "<h4>Invalid Hex</h4>"

        # 2. Struct Panel
        struct_data = {}
        if str(norad_id) == "25397":
            try:
                parsed = GO32_Struct.parse(raw_bytes)
                struct_data = {k: v for k, v in parsed.items() if not k.startswith('_')}
            except Exception as e:
                struct_data = {"error": str(e)}
        else:
            struct_data = {"info": "No struct definition available"}
            
        self.panel_struct.value = f"<h4>Binary Struct (Construct)</h4>{dict_to_html(struct_data)}"

        # 3. Telemetry Panel
        telemetry_dict = {}
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            telem = process_frame(int(norad_id), raw_bytes, "satnogs_db", ts)
            telemetry_dict = telem.to_dict() if telem else {"status": "Decoder Failed (CRC/Validation)"}
        except Exception as e:
            telemetry_dict = {"error": str(e)}
            
        self.panel_telem.value = f"<h4>Standardized (Golden Features)</h4>{dict_to_html(telemetry_dict)}"

    # Events
    def _on_sat_change(self, change):
        if change['new']:
            self.current_norad = change['new']
            self._load_days(change['new'])

    def _on_day_change(self, change):
        if change['new']:
            self._load_frames(self.current_norad, change['new'])

    def _on_frame_change(self, change):
        self.render()

    def _on_prev(self, _):
        if self.frame_slider.value > 0:
            self.frame_slider.value -= 1

    def _on_next(self, _):
        if self.frame_slider.value < self.frame_slider.max:
            self.frame_slider.value += 1

    def show(self):
        # Layout Composition
        controls_top = widgets.HBox([self.sat_dropdown, self.day_dropdown])
        controls_nav = widgets.HBox([self.btn_prev, self.frame_slider, self.btn_next])
        
        panels = widgets.HBox([self.panel_raw, self.panel_struct, self.panel_telem])
        
        ui = widgets.VBox([
            controls_top,
            controls_nav,
            widgets.HTML("<hr>"),
            self.header_label,
            panels
        ])
        display(ui)

# %%
if __name__ == "__main__":
    inspector = TelemetryInspector()
    inspector.show()
