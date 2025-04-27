import plotly.express as px
import pandas as pd

df = pd.read_parquet("./my_data.parquet")

df[["x", "y"]] = pd.DataFrame(df["reduced_embeddings"].tolist(), index=df.index)
fig = px.scatter(x=df["x"], y=df["y"], color=df["cluster_labels"])

fig.write_html("./visualization.html")
