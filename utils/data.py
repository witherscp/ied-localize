"""Data fetchers"""

from os import path
from sys import platform

# check OS
assert platform in "darwin", "linux"

if platform == "darwin":
    NEU_DIR = "/Volumes/Shares/NEU"
elif platform == "linux":
    NEU_DIR = "/shares/NEU"

PROJECTS_DIR = path.join(NEU_DIR, "Projects")
USERS_DIR = path.join(NEU_DIR, "Users")
data_directories = {
    "PROJECTS_DIR": path.join(NEU_DIR, "Projects"),
    "MRI_DIR": path.join(PROJECTS_DIR, "MRI"),
    "DTI_DIR": path.join(PROJECTS_DIR, "DTI"),
    "IED_DIR": path.join(PROJECTS_DIR, "iEEG", "IED_data"),
    "USERS_DIR": path.join(NEU_DIR, "Users"),
    "FIGURES_DIR": path.join(USERS_DIR, "price", "figures"),
    "IED_ANALYSIS_DIR": path.join(USERS_DIR, "price", "ied_analysis")
}