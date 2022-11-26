import json
import os
import re

import einops
import numpy as np
import transformers
import torch
import transformers.models.gpt2.modeling_gpt2
import transformers.models.roberta.modeling_roberta
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import *
from dataset import BlessingDataset
from torch.cuda.amp import GradScaler, autocast
import transformers.generation_utils
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

datasets = {}

def build_model(args):
    if args.task == "generation":
        if "gpt2" in args.model_name:
            model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
        elif "t5" in args.model_name:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        else:
            raise NotImplementedError
    elif args.task == "classification":
        model = transformers.AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)
    else:
        raise NotImplementedError
    if args.mode == "train":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.num_training_steps)
    epoch = 0
    if args.ckpt_path is not None:
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint["model"])
        if args.mode == "train":
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]
    if args.mode == "test":
        return (model, )
    return model, optimizer, scheduler, epoch

def train(args):
    train_dataset = datasets.get("train", datasets.setdefault("train", BlessingDataset(args.train_path, args, is_train=True)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    args.num_training_steps = len(train_dataloader) * args.epochs
    args.num_warmup_steps = int(args.num_training_steps * args.warmup_proportion)
    model, optimizer, scheduler, epoch = build_model(args)
    model.to(args.device)
    # test(args, model=model, epoch=-1)
    for epoch in range(epoch, args.epochs):
        total_loss = 0
        for step, batch in enumerate(iterator := tqdm(train_dataloader, desc=f"Epoch {epoch} Train", total=len(train_dataloader))):
            # if step >= 5:
            #     break
            text = batch.pop("text")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss = loss / args.grad_accum_steps
            loss.backward()
            if (step + 1) % args.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            total_loss = MA(total_loss, loss.item(), step)
            iterator.set_postfix(loss=f"{total_loss:.4f}")
        test(args, model=model, epoch=epoch)
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch}
        torch.save(checkpoint, f"{args.output_path}/epoch={epoch}-loss={total_loss:.4f}.ckpt")

@torch.no_grad()
def test(args, **kwargs):
    test_dataset = datasets.get("test", datasets.setdefault("test", BlessingDataset(args.test_path, args, is_train=False)))
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)
    if (model := kwargs.get("model")) is None:
        model, *_ = build_model(args)
        model.to(args.device)
    model.eval()
    output = []
    epoch = kwargs.get("epoch", None)
    generation_params = {"max_length": args.max_length, "pad_token_id": test_dataset.tokenizer.eos_token_id, "num_return_sequences": args.num_return_sequences, "repetition_penalty": args.repetition_penalty}
    if args.decode_method == "diverse_beam_search":
        generation_params = generation_params | {"num_beams": args.num_beams, "num_beam_groups": args.num_beams, "diversity_penalty": args.diversity_penalty}
    elif args.decode_method == "top_p_sampling":
        generation_params = generation_params | {"top_p": args.top_p, "top_k": args.top_k, "temperature": args.temperature, "do_sample": True}
    else:
        raise NotImplementedError
    all_predictions, all_labels = [], []
    for step, batch in enumerate(tqdm(test_dataloader, desc=f"Test", total=len(test_dataloader))):
        if step >= 2:
            break
        # print(batch)
        text = batch.pop("text")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if args.task == "generation":
            input_ids = batch["input_ids"]
            assert len(input_ids) == 1
            text = {k: v[0] for k, v in text.items()}
            generated_tokens = model.generate(input_ids, **generation_params)
            generated_strings = test_dataset.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            clean = lambda text: re.sub("Send this blessing to (.+?) for (.+?):", "", text).replace("<|endoftext|>", "").strip()
            generated_strings = list(map(clean, generated_strings))
            output.append({"scene": text["Scene"], "object": text["Object"], "prompt": text["Prompt"], "candidates": generated_strings, "references": text["Content"]})
        else:
            logits = model(**batch).logits
            prediction = torch.max(logits, dim=-1).indices
            all_predictions.append(prediction.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            output.extend([{"Content": text["Content"][i], args.cls_target: text[args.cls_target][i], "Prediction": test_dataset.label_list[prediction[i].item()]} for i in range(len(prediction))])

    if args.task == "generation":
        save_file = f"{args.output_path}/{'test' if epoch is None else f'epoch={epoch}'}.json"
        json.dump(output, open(save_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    else:
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        accuracy = accuracy_score(all_labels, all_predictions)
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro")
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        print(f"Epoch {epoch} | Accuracy {accuracy:.4f} | Precision: {avg_precision:.4f} | Recall: {avg_recall:.4f} | F1: {avg_f1:.4f}")
        metrics = {"Average": {"Accuracy": float(accuracy), "Precision": float(avg_precision), "Recall": float(avg_recall), "F1": float(avg_f1)}, "Individual": {test_dataset.label_list[i]: {"Precision": float(precision[i]), "Recall": float(recall[i]), "F1": float(f1[i])} for i in range(len(test_dataset.label_list))}}
        save_file = f"{args.output_path}/{'test' if epoch is None else f'epoch={epoch}'}-acc={accuracy:.4f}-f1={avg_f1:.4f}.json"
        json.dump({"metrics": metrics, "output": output}, open(save_file, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    model.train()

if __name__ == "__main__":
    args = console_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise NotImplementedError
