import pandas as pd
import json
import argparse
import os

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a CSV file to JSON records.")
    parser.add_argument("csv", help="Path to the input CSV file")
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    df = pd.read_csv(csv_path)

    out_path = os.path.splitext(csv_path)[0] + ".json"
    df.to_json(out_path, orient="records", indent=4)

    print(f"Wrote {out_path}")

if __name__ == "__main__":

    main()
