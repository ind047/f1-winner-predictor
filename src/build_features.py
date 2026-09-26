"""Engineer pre-race features for the model from raw results data. Output: data/features.csv.

Leakage rule: every feature for race R uses ONLY races before R (same season).
 """

import pandas as pd

INPUT_FILE = "data/races.csv"
OUTPUT_FILE = "data/features.csv"


# Neutral priors for cold start (round 1 has no season history).

DEFAULT_DRIVER_FORM = 12.0   # midfield-ish average finish
DEFAULT_CONSTRUCTOR_STRENGTH = 2.0   # midfield-ish average finish

def main() -> None:
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(['season','round']).reset_index(drop=True)


    df["driver_form"] = (
        df.groupby(["season","driver_id"])["position_order"]
        .transform(lambda s:s.shift(1).expanding().mean())
    )

    df["constructor_strength"] = (
        df.groupby(['season','constructor'])["points"]
        .transform(lambda s:s.shift(1).expanding().mean())
    )

    # Fill in default values for first race of season (no history).
    df["driver_form"] = df["driver_form"].fillna(DEFAULT_DRIVER_FORM)
    df["constructor_strength"] = df["constructor_strength"].fillna(DEFAULT_CONSTRUCTOR_STRENGTH)

    df["won"] = (df["position_order"] == 1).astype(int)

    keep = [
        "season","round","circuit","driver_id","constructor","grid",
        "driver_form","constructor_strength","won",
    ]

    features = df[keep]

    features.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(features)} rows of features to {OUTPUT_FILE}.")
    print(f"Winners in dataset: {features['won'].sum()} (should be the number of races in the dataset).")

if __name__ == "__main__":
       main()

