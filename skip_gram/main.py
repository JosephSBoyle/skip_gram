"""Skip-gram (word2vec) training on Spanish text.

Notes:
- 2 embeddings per-word: one when word is a context and one for when the word is a target.
    - We can represent words as the sum of these two vectors, or simply throw away the context vector
        and use the target one
- The dot product between two word vectors can be thought of intuitively as the similarity between
    two words. Cosine similarity can be considered a simplified version of this 'true' similarity.
- More out of context ('negative') samples are used during training than in-context ones
- The selection of negative samples has been slightly simplified to be unweighted. 
"""
import collections
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
    word_bag = collections.Counter()
    for line in raw_corpus:
        word_bag.update(line.split())

    vocabulary = {word for word, count in word_bag.items() if count >= min_frequency}
    print(f"Unique words: {len(word_bag)} Vocabulary size: {len(vocabulary)}")
    return vocabulary
class EmbeddingDict(dict):
    """Utility wrapper around a regular dict that allows embedding lookup by word."""
    def __init__(self, embedding_matrix: np.ndarray, word_to_idx: dict[str, int]):
        self._embedding_matrix = embedding_matrix
        self._word_to_idx      = word_to_idx

    def __getitem__(self, __key) -> np.ndarray:
        """Get a word's vector representation."""
        return self._embedding_matrix[self._word_to_idx[__key]]


### Training functions ###
def train_skip_gram(
    file          : Path,
    min_frequency : int   = 100,
    window_size   : int   = 5,
    alpha         : int   = 7,
    embedding_dim : int   = 100,
    η             : float = 1e-3,
) -> None:
    """Train a word-to-vector dictionary using the Skip-gram algorithm from Mikolov et. al."""

    corpus: list[list[str]] = []
    with open(file, "r", encoding="utf-8") as f:
        raw_corpus: list[str] = f.readlines()
    
    vocabulary = _load_spanish_billion_word_corpus_single_file(raw_corpus, min_frequency)
    
    for raw_line in raw_corpus:
        line_words = [word for word in raw_line.split() if word in vocabulary]
        corpus.append(line_words)
    
    vocab_count = len(vocabulary)
    vocabulary  = list(vocabulary)  # Convert to an ordered collection from a set.
    
    word_to_idx = {word : i for i, word in enumerate(vocabulary)}

    ### Randomly initialize word vectors ###
    W = np.random.standard_normal(embedding_dim*vocab_count).reshape((embedding_dim, vocab_count))
    """Target vectors."""
    C = np.random.standard_normal(embedding_dim*vocab_count).reshape((embedding_dim, vocab_count))
    """Context vectors."""
    
    ### Training ###
    #
    # In each step, `j`, we're optimizing a logistic regression classifier to discriminate
    # between words which do and don't occur around a target word in the corpus.
    #
    # We use the weights of this LR classifier to represent the target word. Each time
    # we encounter this target word we do a small update step: which can be intuitively
    # understood as adjusting the words around the target such that they're a little closer
    # together in the embedding space, and a little further away from the randomly selected
    # 'negative' words which do not occur in the same context window.
    #
    # For a more rigorous explanation, I highly recommend 'Speech and Language Processing',
    # the relevant chapter can be found here: 
    # https://web.stanford.edu/~jurafsky/slp3/6.pdf
    for i, line_words in enumerate(corpus):
        for j, target_word in enumerate(line_words):
            # Select context words.
            left_context   = line_words[max(j-window_size, 0) : j]
            right_context  = line_words[j+1: (j+1 + window_size)]
            positive_words = left_context + [target_word] + right_context

            # For simplicity, assume that the probability of sampling a context / target word is 0.
            # It's really (1 + window_size) / |V|.
            #
            # The skip-gram algorithm does a weighted sample with upweighting of rarer words.
            # For simplicity we take an unweighted sample of the entire vocabulary.
            negative_words = random.sample(vocabulary, len(positive_words) * alpha)

            positive_idxs = [word_to_idx[word] for word in positive_words]
            negative_idxs = [word_to_idx[word] for word in negative_words]

            context_positive = C[:, positive_idxs]
            context_negative = C[:, negative_idxs]
            
            a = (_sigmoid(context_positive.T @ W) - 1)
            b = (_sigmoid(context_negative.T @ W))

            # Compute the partial derivative of the classifier loss w.r.t
            # the *context* parameters of the words in each of the two classes.
            dL_by_d_context_positive = a @ W.T
            dL_by_d_context_negative = b @ W.T
            
            # Compute the partial derivative of the classifier loss w.r.t
            # the *target* word parameters.
            dL_by_dw = context_positive @ a + context_negative @ b

            context_positive_update = η * dL_by_d_context_positive
            context_negative_update = η * dL_by_d_context_negative
            target_word_update      = η * dL_by_dw

            # Update the context embedding matrix:
            # move positive words closer to the target word, and negative
            # words further away.
            C[:, positive_idxs] -= context_positive_update.T
            C[:, negative_idxs] -= context_negative_update.T

            # Move the target word's vector closer to the positive context
            # vectors, and further from the negative context vectors.
            W -= target_word_update


        if i % 100 == 0:
            # Log training progress with an example 'king' and 'queen'
            print(f"line {i} of {len(corpus)}")
            rey   = W[:, word_to_idx["rey"]]
            reina = W[:, word_to_idx["reina"]]
            print(f"Dot product of 'rey' and 'reina'            {np.dot(rey, reina):.3f}")
            print(f"Cosine similarity between 'rey' and 'reina' {_cosine_similarity(rey, reina):.3f}")

    # The final embedding matrix can either be `W`, or the sum of `W` and `C`.
    # Let's use the latter - why not ¯_(ツ)_/¯
    embedding_matrix = W + C
    return EmbeddingDict(embedding_matrix, word_to_idx)

if __name__ == "__main__":
    file     = Path("data//spanish_billion_words_00")
    word2vec = train_skip_gram(file)

    print("Dot product between 'boy' and 'girl': %s", word2vec["hijo"] * word2vec["hija"])
    print("Dot product between 'king' and 'queen': %s", word2vec["rey"] * word2vec["reina"])

    boy_to_girl   = word2vec["hijo"] - word2vec["hija"]
    king_to_queen = word2vec["rey"] - word2vec["reina"]

    print("Similarity between the vectors boy->girl and king->queen %s", boy_to_girl * king_to_queen)
