import json
import os

import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset

class BlessingDataset(Dataset):
    def __init__(self, file, args, is_train=True):
        self.is_train = is_train
        self.model_name = args.model_name
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.max_length = args.max_length
        data = pd.read_csv(file)
        self.data = []
        self.task = args.task
        if self.task == "generation":
            if not self.is_train:
                data = data.groupby(["Scene", "Object"]).agg(lambda x: x.tolist()).reset_index()
        elif self.task == "classification":
            self.target = args.cls_target
            if not os.path.exists(args.cls_label_path):
                json.dump({"Scene": sorted(data["Scene"].unique().tolist()), "Object": sorted(data["Object"].unique().tolist())}, open(args.cls_label_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            self.label_list = json.load(open(args.cls_label_path, "r", encoding="utf-8"))[self.target]
            args.num_labels = len(self.label_list)
            self.label_dict = {label: i for i, label in enumerate(self.label_list)}
            clip_samples = args.cls_clip_samples
            data = data.groupby(self.target).apply(lambda df: df.sample(n=min(clip_samples, df.shape[0]))).reset_index(drop=True)
        else:
            raise NotImplementedError
        for i, row in data.iterrows():
            self.data.append(dict(row))
        print(f"Loaded {len(self.data)} Items.")

    def __getitem__(self, index):
        tokenize = lambda text, ast: self.tokenizer.__call__(text=text, max_length=self.max_length, truncation=True, add_special_tokens=ast)
        if self.task == "classification" and "content" not in (item := self.data[index]):
            tokenized_content = tokenize(item["Content"], True)
            self.data[index] = self.data[index] | {"content": tokenized_content, "labels": self.label_dict[item[self.target]]}
        if self.task == "generation" and "prompt" not in (item := self.data[index]):
            # prompt = f"Send this blessing to {item['Object']} for {item['Scene']}:"
            prompt = f"The"
            tokenized_prompt = tokenize(prompt, ast=False)
            self.data[index] = item | {"Prompt": prompt, "prompt": tokenized_prompt}
            if self.is_train:
                tokenized_content = tokenize(item["Content"], ast=True)
                labels = tokenized_content["input_ids"]
                if "gpt2" in self.model_name:
                    tokenized_content["input_ids"] = tokenized_prompt["input_ids"] + tokenized_content["input_ids"] + [self.tokenizer.eos_token_id]
                    tokenized_content["attention_mask"] = tokenized_prompt["attention_mask"] + tokenized_content["attention_mask"] + [1]
                    labels = [-100] * len(tokenized_prompt["input_ids"]) + labels + [self.tokenizer.eos_token_id]
                self.data[index] = self.data[index] | {"content": tokenized_content, "labels": labels}
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_key = "content" if self.task == "classification" or (self.is_train and "gpt2" in self.model_name) else "prompt"
        text_keys = ["Scene", "Object", "Content"]
        if self.task == "generation":
            text_keys += ["Prompt"]
        output = {key: [] for key in ["input_ids", "attention_mask"]}
        if self.is_train or self.task == "classification":
            output["labels"] = []
        output["text"] = {key: [] for key in text_keys}
        for item in batch:
            output["input_ids"].append(item[input_key]["input_ids"])
            output["attention_mask"].append(item[input_key]["attention_mask"])
            if self.is_train or self.task == "classification":
                output["labels"].append(item["labels"])
            for key in text_keys:
                output["text"][key].append(item[key])
        pad = lambda item, value: torch.nn.utils.rnn.pad_sequence(list(map(torch.tensor, item)), batch_first=True, padding_value=value)
        output["input_ids"] = pad(output["input_ids"], 0)
        output["attention_mask"] = pad(output["attention_mask"], 0)
        if self.task == "generation" and self.is_train:
            output["labels"] = pad(output["labels"], -100)
        if self.task == "classification":
            output["labels"] = torch.tensor(output["labels"])
        return output