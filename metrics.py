import functools
import json
import os
import time
import warnings

import gensim
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm

from metrics.bleu import BLEU
from metrics.rouge import RougeL
from metrics.distinct import Distinct
from metrics.ppl import Perplexity
from metrics.wmd import WMD
from metrics.wordcount import WordCount
from metrics.wordsim import WordSim
from metrics.clsscore import ClsScore
from keybert import KeyBERT
from metrics.entanglement import Entanglement
import nltk
nltk.download("stopwords")

metrics = [
    "BLEU",
    "Rouge",
    "Distinct",
    "Perplexity",
    "WMD",
    "WordCount",
    "WordSim",
    "KeySim",
    # "ClsScore",
    "Entanglement"
]

word2vec_model = None
ppl_model = None
ppl_tokenizer = None

def extract_keywords(generation_file, cache_file=None):
    prefix, suffix = generation_file.split(".")
    if cache_file is None:
        cache_file = f"{prefix}-keywords.{suffix}"
    if os.path.exists(cache_file):
        keywords = json.load(open(cache_file, "r", encoding="utf-8"))
        return keywords
    generation_list = json.load(open(generation_file, "r", encoding="utf-8"))
    keywords = []
    warnings.filterwarnings(action="ignore", category=UserWarning)
    key_model = KeyBERT("all-MiniLM-L12-v2")
    for i, item in tqdm(enumerate(generation_list), total=len(generation_list)):
        candidates, references = item["candidates"], item["references"]
        candidates_keywords = key_model.extract_keywords(candidates, keyphrase_ngram_range=(1, 1), top_n=10)
        references_keywords = key_model.extract_keywords(references, keyphrase_ngram_range=(1, 1), top_n=10)
        keywords.append({"candidates": candidates_keywords, "references": references_keywords})
    json.dump(keywords, open(cache_file, "w", encoding="utf-8"), indent=2)
    return keywords

def calculate_metrics(generation_file, labels_file, blessing_words_file, keywords_cache_file=None, output_file=None, **kwargs):
    start = time.time()
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokenizer = transformers.BasicTokenizer(do_lower_case=True)
    if output_file is None:
        prefix, suffix = generation_file.split(".")
        output_file = f"{prefix}-with-metrics.{suffix}"
    generation_list = json.load(open(generation_file, "r", encoding="utf-8"))
    global word2vec_model
    if word2vec_model is None:
        word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("resources/GoogleNews-vectors-negative300.bin", binary=True)
    print("Word2vec model loaded")
    blessing_words = json.load(open(blessing_words_file, "r", encoding="utf-8"))
    bow_occasion = json.load(open("resources/bow/build_bow_occ_2.json", "r", encoding="utf-8"))
    bow_object = json.load(open("resources/bow/build_bow_obj_3.json", "r", encoding="utf-8"))
    if "KeySim" in metrics:
        keysim = WordSim(word2vec_model=word2vec_model)
        keywords = extract_keywords(generation_file, keywords_cache_file)
        assert len(keywords) == len(generation_list)
    if "BLEU" in metrics:
        bleu = BLEU(n_size=4)
    if "Rouge" in metrics:
        rougel = RougeL()
    if "Distinct" in metrics:
        dists = [Distinct(i) for i in range(1, 4)]
        dists_ref = [Distinct(i) for i in range(1, 4)]
    if "Perplexity" in metrics:
        global ppl_model, ppl_tokenizer
        if ppl_model is None:
            ppl_model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
            ppl_tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
        ppl = Perplexity(model=ppl_model, tokenizer=ppl_tokenizer)
    if "WMD" in metrics:
        wmd = WMD(word2vec_model=word2vec_model)
    if "WordCount" in metrics:
        word_count = WordCount()
    if "WordSim" in metrics:
        wordsim = WordSim(word2vec_model=word2vec_model)
    if "ClsScore" in metrics:
        labels = json.load(open(labels_file, "r", encoding="utf-8"))
        scene_dict = {label: i for i, label in enumerate(labels["Scene"])}
        object_dict = {label: i for i, label in enumerate(labels["Object"])}
        cls_scene = ClsScore(ckpt_file="output_scene/epoch=7-loss=0.1089.ckpt", labels_dict=scene_dict)
        cls_object = ClsScore(ckpt_file="output_object/epoch=7-loss=0.9189.ckpt", labels_dict=object_dict)
    if "Entanglement" in metrics:
        entanglement = Entanglement(word2vec_model=word2vec_model, threshold=kwargs.get("threshold", 0.3))
    print(f"Prepare time: {time.time() - start:.2f}s")
    for i, item in tqdm(enumerate(generation_list), total=len(generation_list), desc="Calculating metrics"):
        candidates, references = item["candidates"], item["references"]
        scene, object_ = item["scene"], item["object"]
        tokenize_without_stopwords = lambda sentences: [[token for token in tokenizer.tokenize(sentence) if token not in stopwords] for sentence in sentences]
        candidates_tokens = tokenize_without_stopwords(candidates)
        references_tokens = tokenize_without_stopwords(references)
        @functools.lru_cache(maxsize=100)
        def tokenize(sentence):
            return [token for token in tokenizer.tokenize(sentence) if token not in stopwords]
        scene_tokens = tokenize(scene)
        object_tokens = tokenize(object_)
        for j, (candidate, reference) in enumerate(zip(candidates_tokens, references_tokens)):
            if "BLEU" in metrics:
                bleu.add_inst(candidate, references_tokens)
            if "Rouge" in metrics:
                rougel.add_inst(candidate, references_tokens)
            if "Distinct" in metrics:
                for dist in dists:
                    dist.add_inst(candidate)
                for dist in dists_ref:
                    dist.add_inst(reference)
        if "BLEU" in metrics:
            generation_list[i]["bleu"] = bleu.score()
            bleu.reset()
        if "Rouge" in metrics:
            generation_list[i]["rogue-l"] = rougel.score()
            rougel.reset()
        if "Distinct" in metrics:
            for k in range(1, 4):
                generation_list[i][f"dist-{k}"] = dists[k - 1].score()
                dists[k - 1].reset()
                generation_list[i][f"dist-{k}-ref"] = dists_ref[k - 1].score()
                dists_ref[k - 1].reset()
        if "Perplexity" in metrics:
            generation_list[i]["ppl"] = ppl.evaluate(candidates)
            generation_list[i]["ppl_ref"] = ppl.evaluate(references)
        if "WMD" in metrics:
            generation_list[i]["wmd"] = wmd.evaluate(candidates_tokens, references_tokens)
        if "WordCount" in metrics:
            generation_list[i]["wordcount_blessing"] = word_count.evaluate(candidates_tokens, blessing_words)
            generation_list[i]["wordcount_blessing_ref"] = word_count.evaluate(references_tokens, blessing_words)
        if "WordSim" in metrics:
            generation_list[i]["wordsim_blessing"] = wordsim.evaluate(candidates_tokens, blessing_words)
            # generation_list[i]["wordsim_scene"] = wordsim.evaluate(candidates_tokens, [scene_tokens])
            # generation_list[i]["wordsim_object"] = wordsim.evaluate(candidates_tokens, [object_tokens])
            generation_list[i]["wordsim_blessing_ref"] = wordsim.evaluate(references_tokens, blessing_words)
            # generation_list[i]["wordsim_scene_ref"] = wordsim.evaluate(references_tokens, [scene_tokens])
            # generation_list[i]["wordsim_object_ref"] = wordsim.evaluate(references_tokens, [object_tokens])
        if "KeySim" in metrics:
            def divide(keywords_list):
                keywords, scores = [], []
                keywords.append([])
                scores.append([])
                for item in keywords_list:
                    for keyword, score in item:
                        keywords[-1].append(keyword)
                        scores[-1].append(score)
                return keywords, scores
            candidates_keywords, candidates_weights = divide(keywords[i]["candidates"])
            references_keywords, references_weights = divide(keywords[i]["references"])
            generation_list[i]["keysim_blessing"] = keysim.evaluate(candidates_keywords, blessing_words)
            # generation_list[i]["keysim_scene"] = keysim.evaluate(candidates_keywords, [scene_tokens])
            # generation_list[i]["keysim_object"] = keysim.evaluate(candidates_keywords, [object_tokens])
            generation_list[i]["keysim_blessing_ref"] = keysim.evaluate(references_tokens, blessing_words)
            # generation_list[i]["keysim_scene_ref"] = keysim.evaluate(references_keywords, [scene_tokens])
            # generation_list[i]["keysim_object_ref"] = keysim.evaluate(references_keywords, [object_tokens])
            generation_list[i]["keysim_blessing_weighted"] = keysim.evaluate(candidates_keywords, blessing_words, weights=candidates_weights)
            # generation_list[i]["keysim_scene_weighted"] = keysim.evaluate(candidates_keywords, [scene_tokens], weights=candidates_weights)
            # generation_list[i]["keysim_object_weighted"] = keysim.evaluate(candidates_keywords, [object_tokens], weights=candidates_weights)
            generation_list[i]["keysim_blessing_ref_weighted"] = keysim.evaluate(references_keywords, blessing_words, weights=references_weights)
            # generation_list[i]["keysim_scene_ref_weighted"] = keysim.evaluate(references_keywords, [scene_tokens], weights=references_weights)
            # generation_list[i]["keysim_object_ref_weighted"] = keysim.evaluate(references_keywords, [object_tokens], weights=references_weights)
        if "ClsScore" in metrics:
            generation_list[i]["clsscore_scene"] = cls_scene.evaluate(candidates, scene)
            generation_list[i]["clsscore_object"] = cls_object.evaluate(candidates, object_)
            generation_list[i]["clsscore_scene_ref"] = cls_scene.evaluate(references, scene)
            generation_list[i]["clsscore_object_ref"] = cls_object.evaluate(references, object_)
        if "Entanglement" in metrics:
            if object_ == "general":
                # generation_list[i]["entanglement"] = generation_list[i]["entanglement_ref"] = 0.0
                generation_list[i]["entanglement_penalty"] = generation_list[i]["entanglement_ref_penalty"] = 0.0
                # generation_list[i]["entanglement_norm"] = generation_list[i]["entanglement_ref_norm"] = 0.0
                generation_list[i]["entanglement_norm_penalty"] = generation_list[i]["entanglement_ref_norm_penalty"] = 0.0
                generation_list[i]["entanglement_word_norm_penalty"] = generation_list[i]["entanglement_ref_word_norm_penalty"] = 0.0
            else:
                # generation_list[i]["entanglement"] = entanglement.evaluate(candidates, bow_occasion.get(scene), bow_object.get(object_))
                # generation_list[i]["entanglement_ref"] = entanglement.evaluate(references, bow_occasion.get(scene), bow_object.get(object_))
                # generation_list[i]["entanglement_norm"] = entanglement.evaluate(candidates, bow_occasion.get(scene), bow_object.get(object_), normalized=True)
                # generation_list[i]["entanglement_ref_norm"] = entanglement.evaluate(references, bow_occasion.get(scene), bow_object.get(object_), normalized=True)
                generation_list[i]["entanglement_penalty"] = entanglement.evaluate(candidates, bow_occasion.get(scene), bow_object.get(object_), penalty=1.0)
                generation_list[i]["entanglement_ref_penalty"] = entanglement.evaluate(references, bow_occasion.get(scene), bow_object.get(object_), penalty=1.0, remove_nonpositive=True)
                generation_list[i]["entanglement_norm_penalty"] = entanglement.evaluate(candidates, bow_occasion.get(scene), bow_object.get(object_), normalized="sentence", penalty=1.0)
                generation_list[i]["entanglement_ref_norm_penalty"] = entanglement.evaluate(references, bow_occasion.get(scene), bow_object.get(object_), normalized="sentence", penalty=1.0, remove_nonpositive=True)
                generation_list[i]["entanglement_word_norm_penalty"] = entanglement.evaluate(candidates, bow_occasion.get(scene), bow_object.get(object_), normalized="word", penalty=1.0)
                generation_list[i]["entanglement_ref_word_norm_penalty"] = entanglement.evaluate(references, bow_occasion.get(scene), bow_object.get(object_), normalized="word", penalty=1.0, remove_nonpositive=True)
    json.dump(generation_list, open(output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def summarize(metric_file, output_file=None):
    if output_file is None:
        prefix, suffix = metric_file.split(".")
        output_file = f"{prefix}-summary.csv"
    summary = pd.DataFrame()
    metrics = json.load(open(metric_file, "r", encoding="utf-8"))
    escape_keys = ["prompt", "candidates", "references"]
    for i, metric in enumerate(metrics):
        summary = summary.append({key: value for key, value in metric.items() if key not in escape_keys}, ignore_index=True)
    # print(summary)
    summary = summary[(summary["entanglement_ref_penalty"] > 0.0) | (summary["object"] == "general")]
    summary_without_general = summary[summary["object"] != "general"]
    average_metrics = pd.DataFrame(summary.mean(axis=0)).T
    scene_average_metrics = summary.groupby("scene").mean().reset_index()
    object_average_metrics = summary.groupby("object").mean().reset_index()

    average_metrics_without_general = pd.DataFrame(summary_without_general.mean(axis=0)).T
    scene_average_metrics_without_general = summary_without_general.groupby("scene").mean().reset_index()
    object_average_metrics_without_general = summary_without_general.groupby("object").mean().reset_index()

    summary = pd.concat([average_metrics, average_metrics_without_general, scene_average_metrics, scene_average_metrics_without_general, object_average_metrics, object_average_metrics_without_general, summary], axis=0, ignore_index=True)
    text_cols = ["scene", "object"]
    numeric_cols = [col for col in summary.columns if col not in text_cols]
    summary = summary[[*text_cols, *numeric_cols]].fillna("<AVERAGE>")
    # print(summary)
    print(f"Scene Count: {summary.scene.nunique()}, Object Count: {summary.object.nunique()}")
    summary.to_csv(output_file, index=False)

if __name__ == "__main__":
    output_dirs = ["output", "output_sample", "output_t5", "output_sample_t5", "output-vae-prompt", "output-gan-prompt"]
    file_names = [*(["epoch=3"] * 4), *(["generate-0135060"] * 2)]
    # output_dirs = ["output", "output", "output_sample", "output_t5", "output_sample_t5", "output-vae-prompt", "output-gan-prompt", ]
    # file_name = "epoch=3"
    # file_names = ["news", *(["epoch=3"] * 4), *(["generate-0135060"] * 2)]
    assert len(output_dirs) == len(file_names)
    comparison = pd.DataFrame()
    thresholds = [0.45]
    for i, (output_dir, file_name) in enumerate(zip(output_dirs, file_names)):
        print(output_dir, file_name)
        comparison_part = pd.DataFrame()
        for threshold in thresholds:
            generation_file = f"{output_dir}/{file_name}.json"
            labels_file = "resources/label.json"
            # calculate_metrics(generation_file, labels_file)
            metric_file = f"{output_dir}/{file_name}-with-metrics.json"
            keywords_cache_file = f"{output_dir}/{file_name}-keywords.json"
            blessing_words_file = "resources/bow/blessing_words.json"
            calculate_metrics(generation_file, labels_file=labels_file, keywords_cache_file=keywords_cache_file, output_file=metric_file, blessing_words_file=blessing_words_file, threshold=threshold)
            output_file = f"{output_dir}/{file_name}-with-metrics-summary.csv"
            summarize(metric_file, output_file)
            summary = pd.read_csv(output_file)
            comparison_part = comparison_part.append(summary.loc[1])
        comparison_part["threshold"] = thresholds
        comparison_part["model_path"] = output_dir
        comparison = pd.concat([comparison, comparison_part], axis=0)
    comparison.to_csv("statistics/comparison.csv", index=False)