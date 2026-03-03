from pathlib import Path

import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
OUT = Path("data/telco_churn.csv")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv(OUT, index=False)
    print(f"Saved dataset to {OUT}")


if __name__ == "__main__":
    main()
