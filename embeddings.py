import os
from tqdm import tqdm
from sklearn.datasets import fetch_20newsgroups
import cudf
import cupy
from transformers.models.bert import BertTokenizer, BertModel
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
import torch
from accelerate import Accelerator
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a sample sentence with some stop words.")
filtered_tokens = [token.text for token in doc if not token.is_stop]
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
accelerator = Accelerator()

data = fetch_20newsgroups(subset="all")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer, model = accelerator.prepare(tokenizer, model)


def chunk_document(text, max_length=500, overlap=100):
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = words[i : i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap

    return chunks


def get_bert_embeddings(texts):
    embeddings = embedding_model.encode(texts)

    return embeddings


def clean_document(document):
    doc = document.replace("\n", " ")
    doc = nlp(doc)

    return " ".join([token.text for token in doc if not token.is_stop])


def embed_documents(document):
    chunks = chunk_document(document)
    chunk_embeddings = cupy.array([get_bert_embeddings(chunk) for chunk in chunks])
    document_embedding = cupy.mean(chunk_embeddings, axis=0)
    return document_embedding.flatten()


def main():
    data_file = "./my_data.parquet"

    if not os.path.exists(data_file):
        texts = data.data  # type: ignore
        texts = [clean_document(text) for text in tqdm(texts, desc="Cleaning texts")]

        text_series = cudf.Series(texts, dtype="str")
        print(texts[0])
        # Embed and convert to list (since CuDF doesn't accept cupy arrays)
        embeddings = get_bert_embeddings(texts)

        # Store embeddings as object dtype Series (list of floats)
        embeddings_series = cudf.Series(embeddings)
        df = cudf.DataFrame({"text": text_series, "embeddings": embeddings_series})

        df.to_parquet(data_file)
    else:
        df = cudf.read_parquet(data_file)


if __name__ == "__main__":
    main()
