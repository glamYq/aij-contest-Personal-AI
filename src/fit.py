import argparse
import pickle
from pathlib import Path

import polars as pl

from my_model import MyALSModel


def fit(config) -> None:
    print("Train events:")
    train_events = pl.read_parquet(config.data_dir / "smm_train_events.parquet")
    print(train_events)

    # We do not use 'user_features' and 'user_tags' tables in the baseline,
    # however you can use it in your solution.

    print("User features:")
    print(pl.read_parquet(config.data_dir / "user_features.parquet", n_rows=100))

    print("User tags:")
    print(pl.read_parquet(config.data_dir / "user_tags.parquet", n_rows=100))

    my_model = MyALSModel(n_factors=128, iterations=10, top_k=20)
    my_model.fit(train_events)

    config.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving trained model to {config.model_dir / 'als.pickle'}")
    with open(config.model_dir / "als.pickle", "bw") as f:
        pickle.dump(my_model, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=Path, required=True, help="Path to directory with input data")
    parser.add_argument("-m", "--model_dir", type=Path, required=True, help="Path to model directory with the data need to be passed between fit and predict")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Path to output files with predictions")
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
