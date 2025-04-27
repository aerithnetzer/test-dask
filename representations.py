import cupy as cp
import numpy as np
import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from cuml.dask.common import to_sparse_dask_array
from cuml.dask.naive_bayes import MultinomialNB
import dask
from cuml.feature_extraction.text import (
    TfidfTransformer,
    TfidfVectorizer,
    CountVectorizer,
)


def main():
    cluster = LocalCUDACluster()
    client = Client(cluster)

    df = cudf.read_parquet("./my_data.parquet")

    for i in df.groupby(["cluster_label"]):
        print(i)

    docs = df["text"].to_pandas()

    n_docs = len(docs)

    # 2. Fit cuML CountVectorizer → returns a cupy‐backed sparse CSR matrix
    cv = CountVectorizer()
    X_counts = cv.fit_transform(docs)  # cupyx.scipy.sparse.csr_matrix

    # 3. Compute document frequency (df) per feature:
    #    Turn all non-zero entries to 1, then sum along axis=0.
    X_bool = X_counts.copy()
    X_bool.data = cp.ones_like(X_bool.data)
    # .sum(axis=0) returns a 1×n_features cupy array
    df_gpu = X_bool.sum(axis=0)
    df = cp.asnumpy(df_gpu).ravel()  # bring the df vector to host

    # 4. Compute IDF vector (use the same formula as sklearn’s TfidfTransformer)
    #    idf[t] = log(n_docs / (df[t] + 1)) + 1
    idf = np.log(n_docs / (df + 1)) + 1

    # 5. Compute total term-frequencies per feature (sum of TF across docs)
    tf_sum_gpu = X_counts.sum(axis=0)
    tf_sum = cp.asnumpy(tf_sum_gpu).ravel()

    # 6. Compute *average* TF–IDF (or just weighted sum) for each feature:
    avg_tf = tf_sum / n_docs
    avg_tfidf = avg_tf * idf

    # 7. Extract feature names and get the top-k
    feature_names = cv.get_feature_names()
    # For large vocab, use argpartition for speed:
    top_k = 20
    idx_top = np.argpartition(-avg_tfidf, top_k)[:top_k]
    # Now sort those top_k indices
    idx_top = idx_top[np.argsort(-avg_tfidf[idx_top])]

    # 8. Print your winners

    for i in idx_top:
        print(f"{feature_names[i]}: {avg_tfidf[i]:.4f}")


if __name__ == "__main__":
    main()
