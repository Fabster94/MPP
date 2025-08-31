"""
This module defines several constants used throughout the project.

It provides centralized access to project-wide paths (e.g., data, reports, models)
as well as vocabulary definitions for process steps.

Constants are grouped using NamedTuple objects for convenient attribute-based access.

Examples
--------
>>> from project import constants
>>> constants.PATHS.ROOT  # Path to the project root
>>> constants.VOCAB["bohren"]  # Get token ID for a process step
"""
#standard imports
from pathlib import Path
from collections import namedtuple

# -------------------------------
# Define project-wide filesystem paths
# -----

# Determine ROOT
_ROOT = Path(__file__).parents[2]

# Dictionary of relevant project paths
_path_dict = {
    "ROOT":                 _ROOT,
    "REPORT":       _ROOT / "reports",
    "REPORT_FIGURES":       _ROOT / "reports/figures",
    "CONFIG":               _ROOT / "config",

    "CKPT_DIR":            _ROOT / "src/cadtoseq/ml/models/checkpoints",
    "MODEL_DIR":           _ROOT / "models",

    "PP_DATA":      Path("/home/michelkruse/data_repos/fabricad"), #Productplan (PP) data
    "FEATURE_DATA":           Path("/home/michelkruse/repos/FabriCAD/data/4_feature"),

    "TKMS_FEATURE_DATA":  Path("/workspace/training_data/vecsets")

}

# -------------------------------
# Define token vocabulary
# -------------------------------

# Vocabulary for manufacturing process steps and special tokens
VOCAB = {
    "fräsen": 0,
    "schleifen": 1,
    "bohren": 2,
    "schweißen": 3,
    "drehen": 4,
    "prüfen":5,
    "kontrollieren": 6,
    "START": 7,
    "STOP": 8,
    "PAD": 9,
}
#set the level to which the processes should be dissolved
PROCESS_RESOLUTION_KOMPLEXITY = "coarse"

match PROCESS_RESOLUTION_KOMPLEXITY:
    case "coarse":
        TKMS_VOCAB = {
            "Bohren":0,
            "Drehen":1,
            "Fräsen":2, 
            "START": 3,
            "STOP": 4,
            "PAD": 5,
        }

#check if all keys and values are unique
assert len(VOCAB) == len(set(VOCAB.values())),  ValueError("VOCAB values must be unique")
assert len(VOCAB) == len(set(VOCAB.keys())),  ValueError("VOCAB keys must be unique")

# Inverted vocabulary: maps token IDs back to string labels
INV_VOCAB = {v: k for k, v in VOCAB.items()}
INV_TKMS_VOCAB = {v: k for k, v in VOCAB.items()}

# Convert path dictionary to a namedtuple for attribute-style access
Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)

# clean up for paths constants
del _path_dict
del Paths
del _ROOT

# general clean up
del namedtuple
del Path