import argparse
import pickle
from pathlib import Path

import polars as pl

from my_model import MyALSModel


def make_submission(submission_path: Path, recommendations: pl.DataFrame) -> None:
    print("Converting submission to str format")
    submission = (
        recommendations
        .with_columns(pl.col("item_id").cast(pl.Utf8))
        .groupby("user_id")
        .agg(
            pl.col("item_id")
            .sort_by("score", descending=True)
        )
        .with_columns(
            pl.col("item_id").list.join(" ")
        )
    )

    print("Submission:")
    print(submission)

    print("Write submission to", submission_path)
    submission.write_csv(submission_path)


def predict(config) -> None:
    print(f"Loading trained model from {config.model_dir / 'als.pickle'}:")
    with open(config.model_dir / "als.pickle", "br") as f:
        my_model: MyALSModel = pickle.load(f)

    test_user_ids = pl.read_parquet(config.data_dir / "test_users.parquet")
    recommendations = my_model.predict(test_user_ids["user_id"].to_numpy())

    config.output_dir.mkdir(parents=True, exist_ok=True)
    make_submission(config.output_dir / "result.csv", recommendations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=Path, required=True, help="Path to directory with the input data")
    parser.add_argument("-m", "--model_dir", type=Path, required=True, help="Path to model directory with the data need to be passed between fit and predict")
    parser.add_argument("-o", "--output_dir", type=Path, required=True, help="Path to output files with predictions")
    predict(parser.parse_args())


if __name__ == "__main__":
    main()
