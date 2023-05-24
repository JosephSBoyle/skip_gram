"""
Notes:
- 2 embeddings per-word: one when word is a context and one for when the word is a target.
    - We can represent words as the sum of these two vectors, or simply throw away the context vector
        and use the target one
- dot product between two word vectors ~~ similarity.
    - cosine similarity as a _simplified_ version of this 'true' similarity.
- more negative examples than positive ones (constant factor, alpha)
- words are actually sampled by a function of their frequency plus some bias term which upweights
    rarer words.
"""
from pathlib import Path
from pprint import pprint

import numpy as np
import random; random.seed(7777)

### Utility ###
def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Map x to a probability space [0, 1]"""
    return 1 / (1 + np.exp(-x))

def _cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.inner(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def _load_spanish_billion_word_corpus_single_file(
    raw_corpus    : list[list[str]],
    min_frequency : int
) -> set[str]:
    # 1. Read all the files and construct a vocab of those with `min_frequency` or
    # more occurences. 
    word_bag: dict[str, int] = dict()  # Multiset
    
    for line in raw_corpus:
        words = line.split()
        
        for word in words:
            if word in word_bag:
                word_bag[word] += 1
            else:
                word_bag[word] = 1

    keys_to_drop = []
    for k, v in word_bag.items():
        if v < min_frequency:
            keys_to_drop.append(k)

    for k in keys_to_drop:
        del word_bag[k]

    pprint(word_bag)
    print(len(word_bag))

    vocabulary = word_bag.keys()
    return set(vocabulary)

class EmbeddingDict(dict):
    """Utility wrapper around a regular dict that allows embedding lookup by word."""
    def __init__(self, embedding_matrix: np.ndarray, word_to_idx: dict[str, int]):
        self._embedding_matrix = embedding_matrix
        self._word_to_idx      = word_to_idx

    def __getitem__(self, __key) -> np.ndarray:
        """Get a word's vector representation."""
        return self._embedding_matrix[word_to_idx[__key]]


### Training functions ###
def train_skip_gram(
    file          : Path | None = None,
    min_frequency : int         = 100,
    window_size   : int         = 5,
    alpha         : int         = 7,
    embedding_dim : int         = 100,
    η             : float       = 1e-3,
) -> None:
    # Drop any words with less than `min_frequency`
    #
    # Cycle through the corpus and:
    #     Treat target words and it's neighbors +- `window_size` as positive examples.
    #     Randomly sample alpha * more negative samples than positive ones.
    
    #     Train a LR classifier to discriminate between the negative and positive examples.
    #     The weights of this classifier will be our embedding for the target word.

    corpus: list[list[str]] = []
    with open(file, "r", encoding="utf-8") as f:
        raw_corpus : list[list[str]] = f.readlines()
    
    vocabulary = _load_spanish_billion_word_corpus_single_file(raw_corpus, min_frequency)
    
    for raw_line in raw_corpus:
        line = [word for word in raw_line.split() if word in vocabulary]
        corpus.append(line)
    
    vocab_count = len(vocabulary)
    vocabulary  = list(vocabulary)  # Convert to an ordered collection from a set.
    
    # A poor man's tabular data structure...
    idx_to_word = {i    : word for i, word in enumerate(vocabulary)}
    word_to_idx = {word : i    for i, word in enumerate(vocabulary)}

    ### Randomly initialize word vectors ###
    W = np.random.standard_normal(embedding_dim*vocab_count).reshape((embedding_dim, vocab_count))
    """Target vectors."""
    C = np.random.standard_normal(embedding_dim*vocab_count).reshape((embedding_dim, vocab_count))
    """Context vectors."""
    
    ### Training ###
    for i, line in enumerate(corpus):
        for j, target_word in enumerate(line):
            # get context words
            left_context   = line[max(j-window_size,0) : j]
            right_context  = line[j+1           : (j+1 + window_size)]
            positive_words = left_context + [target_word] + right_context

            # For simplicity, assume that the probability of sampling a context / target word is 0.
            negative_words = random.sample(vocabulary, len(positive_words) * alpha)

            positive_idxs = [word_to_idx[word] for word in positive_words]
            negative_idxs = [word_to_idx[word] for word in negative_words]

            context_positive = C[:, positive_idxs]
            context_negative = C[:, negative_idxs]
            
            a = (_sigmoid(context_positive.T @ W) - 1)
            b = (_sigmoid(context_negative.T @ W))

            dL_by_d_context_positive = a @ W.T
            dL_by_d_context_negative = b @ W.T
            
            dL_by_dw = context_positive @ a + context_negative @ b

            context_positive_update = η * dL_by_d_context_positive
            context_negative_update = η * dL_by_d_context_negative
            target_word_update      = η * dL_by_dw

            # Update the embedding matrices
            C[:, positive_idxs] -= context_positive_update.T
            C[:, negative_idxs] -= context_negative_update.T

            W -= target_word_update


        if i % 100 == 0:
            # Log training progress with an example 'king' and 'queen'
            print(f"line {i} of {len(corpus)}")
            rey   = W[:, word_to_idx["rey"]]
            reina = W[:, word_to_idx["reina"]]
            print(f"Similarity between 'rey' and 'reina'        {np.dot(rey, reina):.3f}")
            print(f"Cosine similarity between 'rey' and 'reina' {_cosine_similarity(rey, reina):.3f}")

    # Return the sum of the target and context matrices!
    embedding_matrix = W + C
    return EmbeddingDict(embedding_matrix, word_to_idx)

if __name__ == "__main__":
    first_file                    = Path("data\\clean_corpus\\spanish_billion_words\\spanish_billion_words_00")
    embedding_matrix, word_to_idx = train_skip_gram(first_file)
