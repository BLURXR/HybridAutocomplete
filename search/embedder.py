from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vocab(p):
    with open(p) as f:
        return [line.strip() for line in f if line.strip()]

def embed(v):
    return model.encode(v, normalize_embeddings=True)

def cosine_search(q, v, vocab_embeddings, top_k=5):
    scores = vocab_embeddings @ embed(q)
    top = np.argsort(-scores)[:top_k]
    return [(v[i], float(scores[i])) for i in top_k]

if __name__ == "__main__":
    k = embed(load_vocab("data/vocab.txt"))

    query = "Java coding"
    results = cosine_search(query, load_vocab("data/vocab.txt"), k)
    for text, score in results:
        print(f"{score:.3f} {text}")