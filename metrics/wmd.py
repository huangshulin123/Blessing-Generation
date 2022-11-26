import gensim
import nltk.corpus
import numpy as np

class WMD:
    def __init__(self, word2vec_file='resources/GoogleNews-vectors-negative300.bin', word2vec_model=None):
        if word2vec_model is None:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            self.word2vec_model = word2vec_model

    def evaluate(self, candidates: list[list[str]], references: list[list[str]]) -> float:
        distances = np.zeros((len(candidates), len(references)))
        for i, candiate in enumerate(candidates):
            for j, reference in enumerate(references):
                distances[i, j] = self.word2vec_model.wmdistance(candiate, reference, norm=True)
        return float(np.mean(distances))

if __name__ == "__main__":
    wmd = WMD(word2vec_file="../resources/GoogleNews-vectors-negative300.bin")
    print(wmd.evaluate([["I", "love", "you"], ["I", "like", "you"]], [["I", "hate", "him"]]))