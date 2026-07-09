from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vocab(path): 
    with open(path) as f: #with means file will close after, name path f
        return [line.strip() for line in f if line.strip()] # returns non empty lines from the file as strings
        #the if line.strip is to filter out empty lines and the strip just removes any whitespace
def embed(v):
    return model.encode(v, normalize_embeddings=True) #uses an input v and returns the embeddings of it in the form
    #in the form of a numpy array, the noramlization meakes sure that embeddings are vectors
    #vectors are normalized to have a length of 1
    # an example what would be returned is 

def cosine_search(q, v, vocab_embeddings, top_k=5): #params are q for query, v for the actual words, vocab_embeddings for the embedded version of the vocab, and top_k for the number of results to return
    q_emb = embed(q) #embed the query into vector
    scores = vocab_embeddings @ q_emb # calculate the cosine sim between query and vocav embeddings, the @ operator is just matrix multiplication, matrix multiplication gives us scores from 0 to 1
    top_idx = np.argsort(-scores)[:top_k] #top index is equal to argsort (means return indices of sorted array) of neg scores, because argsort is ascending, top 5 
    return [(v[i], float(scores[i])) for i in top_idx] #reutns vocab words and their respoective scores for top k results

if __name__ == "__main__": # means that only run is file runs
    vocab = load_vocab("data/vocab.txt") #load it into np array
    k = embed(vocab) #embed the vocab into vector

    query = "Python coding" #test query
    results = cosine_search(query, vocab, k) 
    for text, score in results:
        print(f"{score:.3f} {text}")