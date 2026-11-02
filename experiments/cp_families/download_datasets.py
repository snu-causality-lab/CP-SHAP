"""Download the datasets used by the CP-family experiments into data/downloads/.

Datasets:
    - UCI Concrete           (UCI repository id 165) used by lacp_concrete, cqr_concrete, lacp_calibration_bootstrap
    - Folktables ACS Income  (Census API via folktables) used by mondrian_folktables

Usage:
    python download_datasets.py
"""

import os
import sys

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "downloads"))
os.makedirs(DATA_DIR, exist_ok=True)


def download_concrete() -> None:
    """UCI Concrete Compressive Strength: n=1030, d=8."""
    dest = os.path.join(DATA_DIR, "concrete.csv")
    if os.path.exists(dest):
        print(f"[concrete] already at {dest}")
        return
    print("[concrete] downloading from the UCI repository (id 165) ...")
    try:
        data = fetch_ucirepo(id=165)
        # UCI gives age as int; float columns write the same file as the one used in the paper
        df = pd.concat([data.data.features, data.data.targets], axis=1).astype(float)
        df.columns = [
            "cement", "slag", "fly_ash", "water", "superplasticizer",
            "coarse_agg", "fine_agg", "age", "strength",
        ]
        df.to_csv(dest, index=False)
        print(f"[concrete] saved {len(df)} rows to {dest}")
    except Exception as e:
        sys.exit(f"[concrete] FAILED: {e}")


def download_folktables() -> None:
    """Folktables ACS Income (CA, 2018) via the folktables package."""
    dest = os.path.join(DATA_DIR, "folktables_acs_income_CA.csv")
    if os.path.exists(dest):
        print(f"[folktables] already at {dest}")
        return
    print("[folktables] downloading ACS Income (CA, 2018) ...")
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        sys.exit("[folktables] needs `pip install folktables`")
    try:
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person", root_dir=DATA_DIR
        )
        acs_data = data_source.get_data(states=["CA"], download=True)
        features, label, group = ACSIncome.df_to_numpy(acs_data)
        df = pd.DataFrame(features, columns=ACSIncome.features)
        df["label"] = label
        df["group_race"] = group
        df.to_csv(dest, index=False)
        print(f"[folktables] saved {len(df)} rows to {dest}")
    except Exception as e:
        sys.exit(f"[folktables] FAILED: {e}")


if __name__ == "__main__":
    print(f"target: {DATA_DIR}")
    download_concrete()
    download_folktables()
    print("done.")
