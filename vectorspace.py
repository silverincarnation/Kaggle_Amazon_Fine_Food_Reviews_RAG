import pandas as pd
import ast
import numpy as np
import chromadb
from math import ceil

df = pd.read_csv("Reviews_clean_chunks_emb.csv")
df_val = df.sample(n=5, random_state=1)
df = df.drop(df_val.index)

df["embedding"] = df["embedding"].apply(
    lambda s: np.fromstring(s.strip("[]"), sep=" ").tolist()
)

ids = (df["Id"].astype(str) + "_" + df["chunk_index"].astype(str)).tolist()
embeddings = df["embedding"].tolist()
documents = df["chunk_text"].tolist()

metadatas = [
    {
        "review_id": int(row.Id),
        "product_id": str(row.ProductId),
        "user_id": str(row.UserId),
        "chunk_index": int(row.chunk_index),
        "score": int(row.Score),
        "helpfulness_numerator": int(row.HelpfulnessNumerator),
        "helpfulness_denominator": int(row.HelpfulnessDenominator),
        "time": int(row.Time),
    }
    for row in df.itertuples(index=False)
]

client = chromadb.PersistentClient(path="./chroma_db")
client.delete_collection("amazon_reviews_chunks")
collection = client.get_or_create_collection(name="amazon_reviews_chunks")

batch_size = 2000
n = len(ids)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    collection.add(
        ids=ids[start:end],
        documents=documents[start:end],
        embeddings=embeddings[start:end],
        metadatas=metadatas[start:end],
    )
    print(f"Inserted {end}/{n}")
