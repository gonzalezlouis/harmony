"""Global configuration for music-thermo project.

All paths and constants live here. Edit PROJECT_DIR for your local setup.

Project layout:
    harmony/
      src/   <- this file lives here
      dat/input/   raw datasets
      dat/output/  computed cache (parquet, sequences, npz)
      fig/         figure output
"""
from pathlib import Path

# ------------------------------------------------------------------
# PATHS  (edit these for your machine)
# ------------------------------------------------------------------
ROOT_DIR     = Path(__file__).resolve().parent          # repo root
DATA_DIR     = ROOT_DIR / "data" / "raw"                 # raw datasets (download here)
CACHE_DIR    = ROOT_DIR / "data" / "cache"               # computed records (bundled)
FIG_DIR      = ROOT_DIR / "figures"

# Per-corpus subdirectories
MAESTRO_DIR  = DATA_DIR / "maestro-v3.0.0"
GIANTMIDI_DIR = DATA_DIR / "giantmidi-piano"   # download separately
ASAP_DIR     = DATA_DIR / "asap-dataset"        # download separately

CACHE_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

# ------------------------------------------------------------------
# CHORD STATE SPACE
# ------------------------------------------------------------------
# 12 pitch classes × 5 qualities + 1 "rest" state  =  61 states
PITCH_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC_MAP = {'D-': 'C#', 'E-': 'D#', 'G-': 'F#',
                  'A-': 'G#', 'B-': 'A#', 'C-': 'B', 'F-': 'E'}
QUALITIES    = ['maj', 'min', 'dom7', 'dim', 'aug']
REST_STATE   = 'REST'                              # 61st state

# Radius on the Circle of Fifths layout (used in figures).
RADIUS_MAP   = {'maj': 1.15, 'min': 0.80, 'dom7': 1.60,
                'dim': 1.80, 'aug': 1.40}

# ------------------------------------------------------------------
# ANALYSIS PARAMETERS
# ------------------------------------------------------------------
# Pseudocount for log-ratio regularisation.  Tiny so it does not bias
# bidirectional edges; finite so unidirectional edges contribute.
EPS_PSEUDOCOUNT = 1e-6

# Maximum Markov order for the I(w) saturation curve.
# Beyond w=4 you cannot estimate k-gram counts on a single piece.
W_MAX = 4

# Bootstrap settings
N_BOOTSTRAP = 100
RNG_SEED = 0

# Multiprocessing
N_WORKERS = 4
