import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Compute score from csv file")
    parser.add_argument("file", type=str, help="Path to csv file")
    return parser.parse_args()


def compute_score(file):
    df = pd.read_csv(file)
    squared_average_unpacked = ((1 - df["packedCAP"].mean()) ** 2)
    return squared_average_unpacked



if __name__ == "__main__":
    args = parse_args()
    print(compute_score(args.file))
