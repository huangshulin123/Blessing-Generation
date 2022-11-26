import numpy as np
import nltk

class WordCount:
    def __init__(self):
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def evaluate(self, candidates: list[list[str]], bagofwords: list[str], normalized: bool=True, lemmatized: bool=True) -> float:
        counts = []
        bagofwords = set(bagofwords)
        for candidate in candidates:
            count = 0
            for word in candidate:
                if lemmatized:
                    word = self.lemmatizer.lemmatize(word)
                count += int(word in bagofwords)
            if normalized:
                count = count / len(candidate)
            counts.append(count)
        return float(np.mean(counts))