import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple
from multiprocessing.pool import Pool
from functools import partial
import itertools
from concurrent import futures


class GoldenLDA:
    def __init__(self, timer, name=None):
        self.timer = timer
        self.width = 5
        self.name = "lda"

    def create_document_term_matrix(self, df, col2):
        word_list = self.create_word_list(df, col2)
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(word_list)

    def compute_latent_vectors(self, col2, df) -> np.ndarray:
        document_term_matrix = self.create_document_term_matrix(df, col2)
        transformer = LatentDirichletAllocation(n_components=5, learning_method="online", random_state=99)
        return transformer.fit_transform(document_term_matrix)

    def create_features(self, df, target_cols) -> pd.DataFrame:
        target_cols = target_cols
        col2s = []
        latent_vectors = []

        future_list = list()
        with futures.ProcessPoolExecutor(max_workers=len(target_cols)) as executor:
            for c in target_cols:
                col2s.append(c)
                future_list.append(executor.submit(self.compute_latent_vectors, c, df))
        future_results = [f.result() for f in future_list]
        for res in future_results:
            latent_vectors.append(res.astype(np.float32))
        self.timer.time("done lda ")
        # gc.collect()
        # with Pool(15) as p:
        #     for col1, col2, latent_vector in p.map(
        #             partial(self.compute_latent_vectors, train, test), column_pairs):
        #         col1s.append(col1)
        #         col2s.append(col2)
        #         latent_vectors.append(latent_vector.astype(np.float32))
        gc.collect()
        return self.get_feature(df, col2s, latent_vectors)

    def get_feature(self, df: pd.DataFrame, cs2: List[str], vs: List[np.ndarray]) -> pd.DataFrame:
        card_set = list(set(df["card_id"]))
        features = np.zeros(shape=(len(card_set), len(cs2) * self.width), dtype=np.float32)
        columns = list()
        for i, (col2, latent_vector) in enumerate(zip(cs2, vs)):
            offset = i * self.width
            for j in range(self.width):
                columns.append(self.name + '-' + col2 + '-' + str(j))
            for j, val1 in enumerate(card_set):
                features[j, offset:offset + self.width] = latent_vector[val1]

        ret_df = pd.DataFrame(data=features, columns=columns)
        ret_df["card_id"] = card_set
        return ret_df

    @staticmethod
    def create_word_list(df: pd.DataFrame, col2: str) -> List[str]:
        # col1_size = df["card_id"].max() + 1
        # col2_list = [[] for _ in range(col1_size)]
        # for val2 in df[col2]:
        #     col2_list[val2].append(val2+10)  # 1-9 is a stop word
        # return [' '.join(map(str, a_list)) for a_list in col2_list]

        card_set = list(set(df["card_id"]))
        col2_list = list()
        for val1 in card_set:
            _df = df[df["card_id"] == val1]
            col2_list.append(list(_df[col2]+10))  # add 10 to avoid stop-word

        return [' '.join(map(str, a_list)) for a_list in col2_list]
