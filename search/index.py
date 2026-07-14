import hnswlib
import numpy as np
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))

from embedder import embed, load_vocab

def build_index(vocab_embeddings, dim, ef_construction=200, M=16):
    num_elements = len(vocab_embeddings)
    index = hnswlib.Index(space='ip', dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    ids = np.arange(num_elements)
    index.add_items(vocab_embeddings, ids)
    return index

def set_search_quality(index, ef_search=50):
    index.set_ef(ef_search)

def hnsw_search(query, index, vocab, top_k=5):
    q_emb = embed(query)
    labels, distances = index.knn_query(q_emb, k=top_k)
    results = []
    for idx, dist in zip(labels[0], distances[0]):
        word = vocab[idx]
        similarity = 1 - dist
        results.append((word, similarity))
    return results

def save(index, path):
    index.save_index(path)

def load_index(path, max_elements, dim):
    index = hnswlib.Index(space='ip', dim=dim) 
    index.load_index(path, max_elements = max_elements)
    return index

if __name__ == "__main__":
    vocab = load_vocab("/usr/share/dict/words")
    embeddings = embed(vocab)

    dim = embeddings.shape[1]

    index = build_index(embeddings, dim=dim)
    set_search_quality(index, ef_search=50)

    save(index, "data/vocab_index.bin")

    query = "python coding"
    results = hnsw_search(query, index, vocab, top_k=5)
    for word, score in results:
        print(f"{score:.3f} {word}")