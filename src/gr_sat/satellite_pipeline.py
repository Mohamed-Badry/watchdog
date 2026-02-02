import os
import subprocess
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from typing import List, Dict, Union, Optional

class SatellitePipeline:
    def __init__(self, recordings_dir: Union[str, Path] = "satellite-recordings"):
        """
        Initialize the pipeline.
        
        Args:
            recordings_dir: Path to the directory containing .wav recordings.
                            Can be absolute or relative to current working directory.
        """
        self.recordings_dir = Path(recordings_dir).resolve()
        
        if not self.recordings_dir.exists():
            raise FileNotFoundError(f"Recordings directory not found: {self.recordings_dir}")
            
        if not self.recordings_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.recordings_dir}")
            
        # Manual mapping for high confidence matches based on the file list and gr_satellites list
        self.mapping = {
            "1kuns_pf.wav": "1KUNS-PF",
            "aausat_4.wav": "AAUSAT-4",
            "ao73.wav": "AO-73",
            "at03.wav": "AT03",
            "beesat_9.wav": "BEESAT-9",
            "delphini1.wav": "Delphini-1",
            "entrysat.wav": "EntrySat",
            "floripasat_1.wav": "FloripaSat-1",
            "gomx_3.wav": "GOMX-3",
            "lilacsat1.wav": "LilacSat-1",
            "lilacsat1-image.wav": "LilacSat-1",
            "mysat1.wav": "MYSAT 1",
            "nayif1.wav": "Nayif-1",
            "picsat.wav": "PicSat",
            "qo100.wav": "QO-100",
            "smog_p.wav": "SMOG-P",
            "swiatowid.wav": "Swiatowid",
            "technosat.wav": "TECHNOSAT",
            "trisat.wav": "TRISAT",
            "ty_2.wav": "TY-2",
            "ty_4.wav": "TY 4-01",
            "upsat.wav": "UPSat",
            "vzlusat2.wav": "VZLUSAT-2",
            "zhou_enlai.wav": "Zhou Enlai"
        }

    def get_file_list(self) -> List[str]:
        """Returns a sorted list of .wav files in the recordings directory."""
        return sorted([f.name for f in self.recordings_dir.glob("*.wav")])

    def identify_satellite(self, filename: str) -> str:
        """
        Returns the gr_satellites ID for a given filename.
        """
        if filename in self.mapping:
            return self.mapping[filename]
        
        # Fallback: Try to guess from filename
        name_base = os.path.splitext(filename)[0]
        # Replace underscore with dash as a common variation
        guess = name_base.replace("_", "-").upper()
        return guess

    def load_signal(self, filename: str):
        """
        Loads the WAV file and returns sample rate and normalized float data.
        """
        path = self.recordings_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
            
        fs, data = wavfile.read(path)
        
        # Convert to float and normalize
        if data.dtype != np.float32:
            data = data.astype(np.float32)
            
        # Normalize to +/- 1.0
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data /= max_val
            
        return fs, data

    def decode(self, filename: str, sat_id: Optional[str] = None) -> Dict:
        """
        Runs gr_satellites on the file.
        Returns a dictionary with results.
        """
        if sat_id is None:
            sat_id = self.identify_satellite(filename)
            
        path = self.recordings_dir / filename
        
        # Command: pixi run gr_satellites <sat_id> --wavfile <path>
        # We assume 'pixi' is in path
        cmd = ["pixi", "run", "gr_satellites", sat_id, "--wavfile", str(path)]
        
        print(f"Running decoder: {' '.join(cmd)}")
        
        try:
            # Run with a timeout because some decoders might hang or take long if file is huge
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            return {
                "success": result.returncode == 0,
                "satellite": sat_id,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "satellite": sat_id,
                "stdout": "",
                "stderr": "Timeout expired while decoding.",
            }
        except Exception as e:
            return {
                "success": False,
                "satellite": sat_id,
                "stdout": "",
                "stderr": str(e),
            }
