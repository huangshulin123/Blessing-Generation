import math

import torch
import transformers

class Perplexity:
    def __init__(self, model_name="EleutherAI/gpt-neo-125M", model=None, tokenizer=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is not None and tokenizer is not None:
            self.tokenizer = tokenizer
            self.model = model
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        print("Perplexity model initialized")

    @torch.no_grad()
    def evaluate(self, text: list[str]):
        # text = list(map(lambda x: x + "<|endoftext|>", text))
        # text = list(map(lambda x: "<|endoftext|>" + x + "<|endoftext|>", text))
        batch = self.tokenizer.__call__(text, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False)
        # print(tokenized_text)
        batch = batch.to(self.device)
        labels = batch["input_ids"].clone()
        mask = batch["attention_mask"].bool()
        labels[~mask] = -100
        loss = self.model(**batch, labels=labels).loss
        return math.exp(loss.item())

if __name__ == "__main__":
    ppl = Perplexity()
    val = ppl.evaluate(["At this point, you each deserve an honorary Ph.D in the study of each other.", "You guys are inspiration for anyone looking for a great example of commitment."])
    print(val)