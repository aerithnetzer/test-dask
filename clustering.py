from dask_cuda import LocalCUDACluster
import numpy as np
from dask.distributed import Client
from cuml.dask.cluster import DBSCAN
import cudf
import cupy as cp
import dask.array as da
import dask

dask.config.set({"dataframe.backend": "cudf"})
data_file = "./my_data.parquet"


def main():
    # Start a multi-GPU Dask CUDA cluster
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    # Load the previously saved embeddings
    df = cudf.read_parquet(data_file)
    print(df["text"].iloc[0])
    # Convert list-of-floats column to CuPy array
    embeddings_array = np.array(df["reduced_embeddings"].to_pandas().tolist())
    embeddings_array = cp.asarray(embeddings_array)

    # Create a Dask-CuPy array for distributed computing
    # distributed_X = da.from_array(
    #     embeddings_array, chunks=(500, embeddings_array.shape[1])
    # )

    # Initialize distributed HDBSCAN
    dbscan_model = DBSCAN(eps=0.5, min_samples=10, client=client)

    # Fit and predict cluster labels
    labels = dbscan_model.fit_predict(embeddings_array)
    print("Unique clusters:", cp.unique(labels))
    # Add labels to the DataFrame
    df["cluster_label"] = labels

    # Save the DataFrame with cluster labels
    df.to_parquet(data_file)
    print(df.head())
    print(df.tail())

    print(f"Saved clustered results to '{data_file}'")


if __name__ == "__main__":
    main()
