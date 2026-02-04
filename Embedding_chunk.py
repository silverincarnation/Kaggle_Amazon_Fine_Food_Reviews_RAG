from langchain_text_splitters  import RecursiveCharacterTextSplitter
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
df = pd.read_csv("Reviews_clean.csv")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

def langchain_chunk(text, row_meta, text_splitter):
    chunks = []

    split_texts = text_splitter.split_text(text)

    for idx, chunk_text in enumerate(split_texts):
        chunks.append({
            "Id": row_meta["Id"],
            "ProductId": row_meta["ProductId"],
            "UserId": row_meta["UserId"],
            "chunk_index": idx,
            "HelpfulnessNumerator": row_meta["HelpfulnessNumerator"],
            "HelpfulnessDenominator": row_meta["HelpfulnessDenominator"],
            "Score": row_meta["Score"],
            "Time": row_meta["Time"],
            "chunk_text": chunk_text
        })

    return chunks

chunk_langchain = []
for _, row in df.iterrows():
    chunk_langchain.extend(
        langchain_chunk(
            text=row["full_text"],
            row_meta=row,
            text_splitter=text_splitter
        )
    )

df_chunks = pd.DataFrame(chunk_langchain)
texts = df_chunks["chunk_text"].tolist()
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

df_chunks["embedding"] = list(embeddings)
df_chunks.to_csv("Reviews_clean_chunks_emb.csv", index=False)