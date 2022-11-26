import functools
from typing import Union, Optional
import gensim
import numpy as np
import nltk

class WordSim:
    def __init__(self, word2vec_file='resources/GoogleNews-vectors-negative300.bin', word2vec_model=None):
        if word2vec_model is None:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            self.word2vec_model: gensim.models.KeyedVectors = word2vec_model
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    @functools.lru_cache(maxsize=100)
    def mean_vector_cache(self, words: tuple[str]) -> np.ndarray:
        vectors = [self.word2vec_model.get_vector(word, norm=True) for word in words if word in self.word2vec_model]
        assert len(vectors) > 0, f"No words in {words}"
        return np.mean(np.stack(vectors, axis=0), axis=0)

    def evaluate(self, candidates: list[list[str]], references: list[list[str]], weights: Optional[list[list[float]]]=None, lemmatized: bool=False, aggregator=np.max) -> float:
        assert weights is None or len(weights) == len(candidates), f"Number of candidates and weights must match"
        if lemmatized:
            candidates = list(map(self.lemmatizer.lemmatize, candidates))
            references = list(map(self.lemmatizer.lemmatize, references))
        ref_vectors = np.stack([self.mean_vector_cache(tuple(words)) for words in references], axis=0)
        # cand_vectors = np.zeros((len(candidates), ref_vectors.shape[-1]))
        similarity = np.zeros((len(candidates), len(ref_vectors)))
        for i, candidate in enumerate(candidates):
            word_vectors, word_weights = [], []
            assert weights is None or len(weights[i]) == len(candidates[i]), "Weights must be None or same length as candidates"
            for j, word in enumerate(candidate):
                if not word.isalpha() or word not in self.word2vec_model:
                    continue
                word_vectors.append(self.word2vec_model.get_vector(word, norm=True))
                word_weights.append(weights[i][j] if weights is not None else 1)
            assert len(word_vectors) > 0, f"No words in candidate: {candidate}"
            # cand_vectors[i] = np.mean(np.stack(word_vectors, axis=0), axis=0)
            word_vectors = np.stack(word_vectors, axis=0)
            word_weights = np.array(word_weights) / np.sum(word_weights)
            word_similarity = np.einsum("ij,kj->ik", word_vectors, ref_vectors)
            similarity[i] = np.einsum("i,i->", aggregator(word_similarity, axis=-1), word_weights)
            # cand_vectors[i] = np.einsum("ij,i->j", word_vectors, word_weights) / np.sum(word_weights)
        # print(cand_vectors.shape, ref_vector.shape)
        return float(np.mean(similarity))