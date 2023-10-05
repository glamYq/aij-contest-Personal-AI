from typing import Tuple

import numpy as np
import polars as pl
from tqdm import tqdm
from numpy.typing import NDArray
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix


class MyALSModel:
    def __init__(self, n_factors: int, iterations: int, top_k: int) -> None:
        self._top_k = top_k
        self._als = AlternatingLeastSquares(
            factors=n_factors,
            alpha=40.0,
            random_state=42,
            iterations=iterations,
            calculate_training_loss=True,
        )

    def fit(self, events: pl.DataFrame) -> "MyALSModel":
        print("Encoding user IDs")
        self._user_encoder = MyCategoricalEncoder("user_id", "user_index").fit(events)
        events_encoded = self._user_encoder.transform(events)

        print("Encoding item IDs")
        self._item_encoder = MyCategoricalEncoder("item_id", "item_index").fit(events)
        events_encoded = self._item_encoder.transform(events_encoded)

        print("Creating user-item sparse matrix")
        user_item_matrix = self._crete_user_item_matrix(events_encoded)
        self._user_item_matrix = user_item_matrix

        print()
        print("User count:", user_item_matrix.shape[0])
        print("Item count:", user_item_matrix.shape[1])
        print("Sparsity:", f"{(1 - (user_item_matrix.nnz / np.prod(user_item_matrix.shape))) * 100:.5f}%")
        print()

        print("Run ALS training")
        self._als.fit(user_item_matrix, show_progress=True)

        print("Get top popular items to recommend for cold users")
        self._top_popular_items = (
            events
            .groupby("item_id")
            .agg(pl.count().alias("event_count_per_item"))
            .top_k(self._top_k, by="event_count_per_item")
            .with_columns(
                score=(
                    pl.col("event_count_per_item") /
                    pl.col("event_count_per_item").max()
                ).cast(pl.Float32),
            )
            .drop("event_count_per_item")
        )

        return self


    def predict(self, user_ids: NDArray[np.int32]) -> pl.DataFrame:
        hot_user_ids, cold_user_ids = self._separate_hot_and_cold(user_ids)
        print()
        print("User count:", len(user_ids))
        print("Hot user count:", len(hot_user_ids))
        print("Cold user count:", len(cold_user_ids))
        print()

        hot_users_recommendations = self._predict_hot(hot_user_ids)
        cold_users_recommendations = self._predict_cold(cold_user_ids)

        return pl.concat([hot_users_recommendations, cold_users_recommendations], how="diagonal")


    def _predict_hot(self, hot_user_ids: NDArray[np.int32]) -> pl.DataFrame:
        hot_user_indices = self._user_encoder.transform_array(hot_user_ids)

        n_batches = min(len(hot_user_ids), 10)
        user_id_batches = np.array_split(hot_user_ids, n_batches)
        user_index_batches = np.array_split(hot_user_indices, n_batches)
        result_batches = []

        print("Run ALS prediction")
        for index_batch, id_batch in zip(tqdm(user_index_batches), user_id_batches):
            item_indices, item_scores = self._als.recommend(
                index_batch,
                self._user_item_matrix[index_batch],
                filter_already_liked_items=True,
                N=self._top_k,
            )

            result_batch = pl.DataFrame({
                "user_id": id_batch,
                "item_index": item_indices,
                "score": item_scores,
            })
            result_batches.append(result_batch)

        print("Repacking hot users recommendations")
        result_grouped = pl.concat(result_batches)
        del result_batches
        result = result_grouped.explode(["item_index", "score"])

        print("Decoding hot users recommendations")
        result_decoded = self._item_encoder.inverse_transform(result)

        return result_decoded


    def _predict_cold(self, cold_user_ids: NDArray[np.int32]) -> pl.DataFrame:
        print("Fill cold user recommendations with top popular items")

        popular_items = self._top_popular_items.head(self._top_k)
        item_ids = popular_items["item_id"].to_numpy()
        scores = popular_items["score"].to_numpy()

        result = pl.DataFrame({
            "user_id": np.repeat(cold_user_ids, self._top_k),
            "item_id": np.tile(item_ids, len(cold_user_ids)),
            "score": np.tile(scores, len(cold_user_ids)),
        })

        return result


    def _separate_hot_and_cold(self, user_ids: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        user_ids_df = pl.Series("user_id", user_ids).to_frame()
        known_user_ids_df = self._user_encoder.mapping[["user_id"]]

        hot_user_ids = user_ids_df.join(known_user_ids_df, on="user_id", how="inner")["user_id"].to_numpy()
        cold_user_ids = user_ids_df.join(known_user_ids_df, on="user_id", how="anti")["user_id"].to_numpy()

        return hot_user_ids, cold_user_ids


    def _crete_user_item_matrix(self, encoded_events: pl.DataFrame) -> csr_matrix:
        user_indices = encoded_events["user_index"].to_numpy()
        item_indices = encoded_events["item_index"].to_numpy()

        user_item_matrix = csr_matrix(
            (
                np.ones(len(encoded_events), dtype=np.float32),
                (
                    user_indices,
                    item_indices,
                ),
            ),
            shape=(user_indices.max() + 1, item_indices.max() + 1),
            dtype=np.float32,
        )

        return user_item_matrix


class MyCategoricalEncoder:
    def __init__(self, source_column: str, target_column: str) -> None:
        self.source_column = source_column
        self.target_column = target_column

    @property
    def mapping(self) -> pl.DataFrame:
        return self._mapping

    def fit(self, df: pl.DataFrame) -> "MyCategoricalEncoder":
        self._mapping = (
            df[[self.source_column]]
            .unique(maintain_order=True)
            .with_row_count(self.target_column)
            .with_columns(pl.col(self.target_column).cast(pl.Int32))
        )
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        encoded_df = df.join(self._mapping, on=self.source_column).drop(self.source_column)
        return encoded_df

    def transform_array(self, values: NDArray) -> NDArray[np.int32]:
        df = pl.Series(self.source_column, values).to_frame()
        encoded_df = self.transform(df).drop_nulls()
        return encoded_df[self.target_column].to_numpy()

    def inverse_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        decoded_df = df.join(self._mapping, on=self.target_column).drop(self.target_column)
        return decoded_df
