from dask_cuda import LocalCUDACluster
import numpy as np
from dask.distributed import Client
from cuml.dask.manifold import UMAP as MNMG_UMAP
from cuml.manifold import UMAP
import dask.array as da
import cupy
import cudf


def main():
    df = cudf.read_parquet("./my_data.parquet")
    df_selection = df.sample(n=100)

    cluster = LocalCUDACluster(threads_per_worker=4)
    client = Client(cluster)

    local_model = UMAP(n_components=2)
    distributed_model = MNMG_UMAP(model=local_model)

    embeddings_array_sample = np.array(df_selection["embeddings"].to_pandas().tolist())
    embeddings_array = np.array(df["embeddings"].to_pandas().tolist())
    embeddings_array = cupy.asarray(embeddings_array)

    local_model.fit(embeddings_array_sample)
    distributed_X = da.from_array(
        embeddings_array, chunks=(1000, embeddings_array.shape[1])
    )
    embedding = distributed_model.transform(distributed_X)

    result = embedding.compute()

    reduced_embeddings = cupy.asnumpy(result).tolist()

    df["reduced_embeddings"] = reduced_embeddings

    df.to_parquet("./my_data.parquet")

    print(result)


if __name__ == "__main__":
    main()
