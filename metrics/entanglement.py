import functools
from collections import Counter

import gensim
import nltk
import numpy as np
import transformers


class Entanglement:
    def __init__(self, word2vec_file='resources/GoogleNews-vectors-negative300.bin', word2vec_model=None, threshold=0.3):
        if word2vec_model is None:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        else:
            self.word2vec_model = word2vec_model
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.threshold = threshold
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.tokenizer = transformers.BasicTokenizer(do_lower_case=True)
        self.puncs = ",.!?;"

    def get_bow(self, bow):
        vectors = []
        for word in bow:
            word = word.lower()
            if word in self.word2vec_model:
                vectors.append(self.word2vec_model.get_vector(word, norm=True))
            else:
                word = self.lemmatizer.lemmatize(word.lower())
                if word in self.word2vec_model:
                    vectors.append(self.word2vec_model.get_vector(word, norm=True))
        if len(vectors) == 0:
            return None
        else:
            return np.stack(vectors, axis=0)

    @functools.lru_cache(maxsize=100)
    def get_bow_cached(self, bow: tuple[str]):
        return self.get_bow(bow)

    def evaluate(self, candidates: list[str], bow_occasion: list[str], bow_object: list[str], normalized=None, penalty=0.0, remove_nonpositive=False):
        if bow_occasion is None or bow_object is None:
            return 0
        bow_occasion = self.get_bow_cached(tuple(bow_occasion))
        bow_object = self.get_bow_cached(tuple(bow_object))
        score = 0
        valid_count = 0
        for candidate in candidates:
            candidate_tokens = [token for token in self.tokenizer.tokenize(candidate) if token not in self.stopwords]
            if candidate_tokens[-1] not in self.puncs:
                candidate_tokens.append(".")
            sentences = []
            start = 0
            for end, token in enumerate(candidate_tokens):
                if token in self.puncs:
                    sentences.append(candidate_tokens[start:end+1])
                    start = end + 1
            sentence_score = self.evaluate_one_sentence(sentences, bow_occasion, bow_object, normalized, penalty)
            if remove_nonpositive and sentence_score <= 0:
                continue
            score += sentence_score
            valid_count += 1
        return score / (valid_count + 1e-10)

    def evaluate_one_sentence(self, tokenized_content, bow_occasion, bow_object, normalized=None, penalty=0.0):
        score = 0
        counter_occ = Counter()
        counter_obj = Counter()
        words_count = 0
        for semi_sent_words in tokenized_content:
            words_count += len(semi_sent_words)
            semi_sent = self.get_bow(semi_sent_words)
            if semi_sent is None:
                continue
            similarity_occ = np.einsum("ij,kj->ki", bow_occasion, semi_sent)
            similarity_obj = np.einsum("ij,kj->ki", bow_object, semi_sent)
            out_occ = similarity_occ.max(axis=1)
            out_obj = similarity_obj.max(axis=1)
            lis_occ = np.where(out_occ >= self.threshold)[0]
            lis_obj = np.where(out_obj >= self.threshold)[0]
            words_occ = Counter([semi_sent_words[i] for i in lis_occ])
            counter_occ = counter_occ + words_occ
            words_obj = Counter([semi_sent_words[i] for i in lis_obj])
            counter_obj = counter_obj + words_obj
            count_occ = len(lis_occ)
            count_obj = len(lis_obj)

            score += min(count_obj, count_occ)
            flag = 0
            for inum in range(0, len(semi_sent)):
                if inum in lis_obj and inum in lis_occ:
                    score += 1.0
                    flag = 3
                elif inum in lis_obj:
                    if flag in [2, 3]:
                        score += 1.0
                    flag = 1
                elif inum in lis_occ:
                    if flag in [1, 3]:
                        score += 1.0
                    flag = 2
        for count in counter_occ.values():
            score = score - max(0.0, penalty * (count - 2))
        for count in counter_obj.values():
            score = score - max(0.0, penalty * (count - 2))
        if normalized == "sentence":
            score = score / len(tokenized_content)
        elif normalized == "word":
            score = score / words_count
        return score
