import pandas as pd

def clean_data(path):

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("%", "pct", regex=False)
    )

    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")],
                 errors="ignore")

    df = df[(df["year"] >= 2019) & (df["year"] <= 2023)].copy()

    cols_needed = [
        "state","year","ev_share_pct",
        "stations","total_vehicles",
        "per_cap_income","bachelor_attainment",
        "gasoline_price_per_gallon",
        "price_cents_per_kwh",
        "trucks_share","incentives"
    ]

    df = df[cols_needed]
    df = df[df["state"] != "district_of_columbia"]

    df["stations_per_100k_vehicles"] = (
        df["stations"]/df["total_vehicles"]*100000
    )

    df["per_cap_income"] /= 1000

    df = df.drop(columns=["stations","total_vehicles"])
    df = df.dropna()

    return df
