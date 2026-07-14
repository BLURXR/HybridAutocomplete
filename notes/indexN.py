import hnswlib
import numpy as np
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'search'))

#from embedder import embed, load_vocab

def build_index(vocab_embeddings, dim, ef_construction=200, M=16): #16 neighbors per node, less dead spots, also ef_construction of 200 to eval 200 nodes before placement (200 nodes in shortlist)
    num_elements = len(vocab_embeddings)
    index = hnswlib.Index(space='ip', dim=dim) #ip means inner product, we set space to this because vectors are normalized from embed func
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M) # build time params and allocate graph
    ids = np.arange(num_elements) #create array of id for each vocab (map back to the word)
    index.add_items(vocab_embeddings, ids) #placing vectors
    return index

def set_search_quality(index, ef_search=50): #shortlist of 50 canidates
    index.set_ef(ef_search) #set param

def hnsw_search(query, index, vocab, top_k=5):
    q_emb = embed(query) #embed the query
    labels, distances = index.knn_query(q_emb, k=top_k) #get the top k results from the index
    # 1 - ip means distance so less is closer
    results = []
    for idx, dist in zip(labels[0], distances[0]):
        word = vocab[idx]
        similarity = 1 - dist
        results.append((word, similarity))
    return results

def save(index, path):
    index.save_index(path)

def load_index(path, max_elements, dim):
    index = hnswlib.Index(space='ip', dim=dim) #max elements greater than orgiginal or you cant add more items later
    index.load_index(path, max_elements = max_elements)
    return index

if __name__ == "__main__":
    vocab = load_vocab("/usr/share/dict/words")
    vocab = random.sample(vocab, 5000)  # sample random 5000 words for testing
    embeddings = embed(vocab)

    # derive dim dynamically instead of hardcoding 384 — safer if model ever changes
    dim = embeddings.shape[1]

    index = build_index(embeddings, dim=dim)
    set_search_quality(index, ef_search=50)

    save(index, "data/vocab_index.bin")

    query = "python coding" #intrestingly for this query top results are relating to python but not coding, but this is fine as humans type sequentially
    #later on, when for example a human types a full sentence query, we will need to use model on sentences
    results = hnsw_search(query, index, vocab, top_k=5)
    for word, score in results:
        print(f"{score:.3f} {word}")
