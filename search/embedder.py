from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vocab(path): 
    with open(path) as f:
        return [line.strip() for line in f if line.strip()] 
        
def embed(v):
    return model.encode(v, normalize_embeddings=True) 

def cosine_search(q, v, vocab_embeddings, top_k=5): 
    q_emb = embed(q) 
    scores = vocab_embeddings @ q_emb 
    top_idx = np.argsort(-scores)[:top_k] 
    return [(v[i], float(scores[i])) for i in top_idx]

if __name__ == "__main__": 
    vocab = load_vocab("data/vocab.txt") 
    k = embed(vocab) 

    query = "Python coding" 
    results = cosine_search(query, vocab, k) 
    for text, score in results:
        print(f"{score:.3f} {text}")