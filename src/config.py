from pathlib import Path

ROOT_DIR     = Path(__file__).resolve().parent
DATA_DIR     = ROOT_DIR / "data" / "raw"
CACHE_DIR    = ROOT_DIR / "data" / "cache"
FIG_DIR      = ROOT_DIR / "figures"

MAESTRO_DIR  = DATA_DIR / "maestro-v3.0.0"
GIANTMIDI_DIR = DATA_DIR / "giantmidi-piano"
ASAP_DIR     = DATA_DIR / "asap-dataset"

CACHE_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

PITCH_NAMES  = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC_MAP = {'D-': 'C#', 'E-': 'D#', 'G-': 'F#',
                  'A-': 'G#', 'B-': 'A#', 'C-': 'B', 'F-': 'E'}
QUALITIES    = ['maj', 'min', 'dom7', 'dim', 'aug']
REST_STATE   = 'REST'

RADIUS_MAP   = {'maj': 1.15, 'min': 0.80, 'dom7': 1.60,
                'dim': 1.80, 'aug': 1.40}

EPS_PSEUDOCOUNT = 1e-6

W_MAX = 4

N_BOOTSTRAP = 100
RNG_SEED = 0

N_WORKERS = 4
