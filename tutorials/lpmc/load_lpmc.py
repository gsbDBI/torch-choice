"""Loader script for the London Passenger Mode Choice (LPMC) dataset.

The raw ``lpmc.dat`` file (tab-separated, ~13 MB, 81,086 rows) is hosted by
the EPFL ChoiceModels MOOC on edX and used by Bierlaire's `mooc-discrete-
choice` reference notebook. We don't redistribute the file here; we download
it on first use and cache it next to this script.

Citation (required by Hillel's technical report):

    Hillel, T., Elshafie, M. Z. E. B., Jin, Y. (2018). "Recreating passenger
    mode choice-sets for transport simulation: A case study of London, UK."
    *Proceedings of the Institution of Civil Engineers - Smart Infrastructure
    and Construction*, 171(1), 29-42.

Usage:

    from load_lpmc import load_lpmc
    df = load_lpmc()                 # downloads + caches lpmc.csv
    df = load_lpmc(force=True)       # force re-download
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

# Public EPFL course-asset URL used by Bierlaire's MOOC notebook
# (https://github.com/michelbierlaire/mooc-discrete-choice/blob/master/LPMC_DCM_ML.ipynb).
# The file is the canonical Hillel et al. 2018 LPMC release in tab-separated
# format. Mirror it locally on first use; we don't redistribute it.
LPMC_URL = (
    "https://courses.edx.org/"
    "asset-v1:EPFLx+ChoiceModels2x+3T2021+type@asset+block@lpmc.dat"
)
HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "lpmc.dat"
CSV_PATH = HERE / "lpmc.csv"


def load_lpmc(*, force: bool = False) -> pd.DataFrame:
    """Return the LPMC dataset as a DataFrame; download + cache if needed.

    The CSV is comma-separated for portability; the original ``lpmc.dat`` is
    tab-separated. Both are kept side-by-side for transparency.
    """
    if force or not CSV_PATH.exists():
        if force or not RAW_PATH.exists():
            print(f"[load_lpmc] downloading {LPMC_URL} -> {RAW_PATH}")
            urllib.request.urlretrieve(LPMC_URL, RAW_PATH)
        df = pd.read_table(RAW_PATH, sep="\t")
        df.to_csv(CSV_PATH, index=False)
        print(f"[load_lpmc] wrote {CSV_PATH} ({df.shape[0]} rows x {df.shape[1]} cols)")
    return pd.read_csv(CSV_PATH)


if __name__ == "__main__":
    df = load_lpmc()
    print(df.shape)
    print(df["travel_mode"].value_counts().sort_index())
