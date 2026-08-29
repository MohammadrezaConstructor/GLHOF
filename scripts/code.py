import pandas as pd

files = {
    2015: "data/raw/can_2015.csv",
    2016: "data/raw/can_2016.csv",
    2017: "data/raw/can_2017.csv",
}

for year, path in files.items():
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        usecols=["DT_DISPATCH", "DT_AWARD"],
        nrows=30,
    )

    print("\n", "=" * 60)
    print(year)
    print("=" * 60)

    print("\nDT_DISPATCH examples:")
    print(df["DT_DISPATCH"].dropna().unique()[:10])

    print("\nDT_AWARD examples:")
    print(df["DT_AWARD"].dropna().unique()[:10])