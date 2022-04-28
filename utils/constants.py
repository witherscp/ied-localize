from pathlib import Path
from sys import platform

# check OS
assert platform in "darwin", "linux"

if platform == "darwin":
    NEU_DIR = Path("/Volumes/Shares/NEU")
elif platform == "linux":
    NEU_DIR = Path("/shares/NEU")

PROJECTS_DIR = NEU_DIR / "Projects"
USERS_DIR = NEU_DIR / "Users"
data_directories = {
    "PROJECTS_DIR": NEU_DIR / "Projects",
    "MRI_DIR": PROJECTS_DIR / "MRI",
    "DTI_DIR": PROJECTS_DIR / "DTI",
    "IED_DIR": PROJECTS_DIR / "iEEG" / "IED_data",
    "USERS_DIR": NEU_DIR / "Users",
    "FIGURES_DIR": USERS_DIR / "price" / "figures",
    "IED_ANALYSIS_DIR": USERS_DIR / "price" / "ied_analysis"
}

# propagation parameters
MAX_WM_VEL = 3
MIN_WM_VEL = 1.2

MAX_GEO_VEL = 0.45
MIN_GEO_VEL = 0.25

# Roman numeral conversion list
NUM_MAP = [(10, 'X'), 
           (9, 'IX'), 
           (5, 'V'), 
           (4, 'IV'), 
           (1, 'I')
        ]