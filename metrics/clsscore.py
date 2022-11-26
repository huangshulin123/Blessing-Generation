from typing import Union

import torch
import transformers


class ClsScore:
    def __init__(self, ckpt_file, labels_dict, model_name="roberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels_dict))
        self.model.load_state_dict(torch.load(ckpt_file)["model"])
        self.model.to(self.device)
        self.model.eval()
        self.labels_dict = labels_dict
        print("Classifier loaded from {}".format(ckpt_file))

    @torch.no_grad()
    def evaluate(self, candidates: list[str], label: str) -> float:
        label_index = self.labels_dict[label]
        candidates_tokens = self.tokenizer.__call__(text=candidates, max_length=512, truncation=True, add_special_tokens=True, padding=True, return_tensors="pt")
        candidates_tokens = candidates_tokens.to(self.device)
        logits = self.model(**candidates_tokens).logits
        probs = torch.softmax(logits, dim=-1)
        label_probs = probs[:, label_index]
        return label_probs.mean().item()
