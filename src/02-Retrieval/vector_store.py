import faiss
import numpy as np
import pickle


def create_index(embeddings):
    dimension = embeddings.shape[1]

    # Inner Product on normalized vectors = Cosine Similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


def search_index(index, query_embedding, top_k=5):

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(
        query_embedding
    )

    query_vector = np.array(
        [query_embedding]
    ).astype("float32")

    similarities, indices = index.search(
        query_vector,
        top_k
    )

    return similarities[0], indices[0]


def save_index(index, chunks):

    faiss.write_index(
        index,
        "index/faiss.index"
    )

    with open(
        "index/chunks.pkl",
        "wb"
    ) as file:
        pickle.dump(chunks, file)


def load_index():

    index = faiss.read_index(
        "index/faiss.index"
    )

    with open(
        "index/chunks.pkl",
        "rb"
    ) as file:
        chunks = pickle.load(file)

    return index, chunks