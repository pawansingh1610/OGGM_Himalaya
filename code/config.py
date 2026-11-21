"""Machine-specific locations. Set these environment variables (or edit the defaults) before running anything.

HMA_OGGM_ROOT   directory that holds download_cache/ (third-party input data, NOT in this archive),
                calibration_output_v3/ and compiled_v3_<run>/.   default: ~/OGGM
HMA_CODE_DIR    directory holding code/pipeline/ (the model code).   default: <this archive>/code/pipeline
HMA_PYTHON      python interpreter of the conda env 'oggm' (used by the *.sh chains).  default: python
HMA_SENS_CODE_DIR  directory with the sensitivity-run copy of the pipeline (code/pipeline/provenance).
"""
import os
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parent.parent
OGGM_ROOT = Path(os.environ.get("HMA_OGGM_ROOT", Path.home() / "OGGM"))
CODE_DIR = Path(os.environ.get("HMA_CODE_DIR", ARCHIVE / "code" / "pipeline"))
SENS_CODE_DIR = Path(os.environ.get("HMA_SENS_CODE_DIR", ARCHIVE / "code" / "pipeline" / "provenance"))
DOWNLOAD_CACHE = Path(os.environ.get("HMA_DOWNLOAD_CACHE", OGGM_ROOT / "download_cache"))
PYTHON = os.environ.get("HMA_PYTHON", "python")
