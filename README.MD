# Pump Curve Digitizer

A professional tool for digitizing pump performance curves from PDF catalogs.  
Designed for engineers and pump specialists to extract Q‑H, NPSH, and efficiency data and store them in a structured JSON format for further use (e.g., in a pump selection application).

## Features

- **Interactive graph area selection** – crop the region of interest.
- **Two‑point axis calibration** – define real‑world values for X (flow), left Y (head), right Y (NPSH), and efficiency axis.
- **Curve digitization** – click points along Q‑H curves, optional NPSH curves, and efficiency points.
- **Polynomial fitting**:
  - Q‑H and NPSH: cubic polynomials (degree 3).
  - Efficiency: quartic polynomials (degree 4) – fitted only on user‑provided points, with a valid range.
- **Multi‑page PDF support** – process several pages in one session; new impeller diameters are merged into the same pump entry.
- **JSON catalog** – appends data to a pump catalog, merging by model + rpm, adding only new diameters.
- **Self‑contained script** – uses [PEP 723](https://peps.python.org/pep-0723/) metadata for automatic dependency management with `uv`.

## Requirements

- Python 3.9 or higher
- [poppler](https://poppler.freedesktop.org/) (for `pdf2image`) – install via system package manager:
  - Ubuntu/Debian: `sudo apt install poppler-utils`
  - macOS: `brew install poppler`
  - Windows: download from [poppler for Windows](http://blog.alivate.com.au/poppler-windows/) and add `bin` folder to PATH.

## Installation & Usage

### Option 1: Using `uv` (recommended)

1. Install `uv` (if not already):
   ```bash
   pip install uv
   # or
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Clone the repository and run the script:

  ```bash
  git clone https://github.com/yourusername/pump-curve-digitizer.git
  cd pump-curve-digitizer
  uv run pump_digitizer.py path/to/your/catalog.pdf
  uv will automatically create a virtual environment and install all required dependencies listed in the script's metadata.
  ```
Option 2: Traditional pip
Clone the repository:

  ```bash
  git clone https://github.com/yourusername/pump-curve-digitizer.git
  cd pump-curve-digitizer
  Create and activate a virtual environment (optional but recommended):
  ```
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
Install dependencies manually:

  ```bash
  pip install opencv-python numpy pandas matplotlib pdf2image scipy
  ```
Run the script:

  ```bash
  python pump_digitizer.py path/to/your/catalog.pdf
  ```
How to Use (Step‑by‑Step)
Select graph area – drag a rectangle around the pump curves.

Calibrate axes – click two known points for each axis (flow, head, NPSH, efficiency) and enter their real values.
NPSH and efficiency axes are optional – you can skip them by pressing s when prompted.

Enter pump information – model, rated speed (rpm), and whether impeller trimming is allowed.

Digitize curves – for each impeller diameter:

Click points along the Q‑H curve (press c when done).

Optionally digitize the NPSH curve (if present).

Optionally click on the Q‑H curve where efficiency is known and enter the efficiency value.

Repeat – after finishing one diameter, the script asks if there are more diameters.

Export – choose CSV, JSON, or both. The JSON file acts as a growing catalog: if the same pump model + rpm already exists, new impeller diameters are added without duplication.

Plot – optionally view the fitted curves and original points.

Output Format (JSON)
The JSON catalog has the following structure:

json
Structure:
    {
        "pumps": [
            {
                "model": "...",
                "rpm": 2980,
                "type": "single" or "multi",
                "trim_allowed": true/false (only for single-stage),
                "impeller_options": [
                    {
                        "variant": {"type": "diameter", "value": 210},  # universal
                        "diameter": 210,                                 # for backward compatibility
                        "stages": null,
                        "curves": {
                            "q_h": {"coefficients": [...], "q_min": ..., "q_max": ...},
                            "npsh": {...}   (optional)
                        },
                        "efficiency": {                                 # optional
                            "coefficients": [...],
                            "rmse": ...,
                            "q_min": ...,
                            "q_max": ...
                        }
                    },
                    ...
                ]
            }
        ]
    }
Example
After processing a PDF page with curves for a pump at 2980 rpm and diameters 210 and 190 mm, you'll get a JSON file containing both impeller options under the same pump entry.

Contributing
Feel free to open issues or submit pull requests for improvements or bug fixes.

License
MIT (or your preferred license)