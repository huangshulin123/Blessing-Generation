import argparse

MA = lambda old, new, step: old * step / (step + 1) + new / (step + 1)

def console_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--model_name", type=str, default="gpt2", help="model name")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--test_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--max_length", type=int, default=200, help="max sequence length")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup proportion")
    parser.add_argument("--device", type=str, default="cuda", help="device")

    parser.add_argument("--decode_method", type=str, default="diverse_beam_search", help="decode method", choices=["diverse_beam_search", "top_p_sampling"])
    parser.add_argument("--top_p", type=float, default=0.95, help="top p")
    parser.add_argument("--top_k", type=int, default=100, help="top k")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--num_beams", type=int, default=20, help="number of beams")
    parser.add_argument("--num_return_sequences", type=int, default=20, help="number of return sequences")
    parser.add_argument("--diversity_penalty", type=float, default=1.0, help="diversity penalty")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="repetition penalty")

    parser.add_argument("--clip_norm", type=float, default=5.0, help="clip norm")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="gradient accumulation steps")

    parser.add_argument("--mode", type=str, default="train", help="train or test", choices=["train", "test"])
    parser.add_argument("--task", type=str, default="generation", help="generation or classification", choices=["generation", "classification"])
    parser.add_argument("--cls_target", type=str, default="Scene", help="classification target", choices=["Scene", "Object"])
    parser.add_argument("--cls_clip_samples", type=int, default=3000, help="classification samples")
    parser.add_argument("--seed", type=int, default=19260817, help="random seed")
    parser.add_argument("--train_path", type=str, default="resources/train.csv", help="train path")
    parser.add_argument("--test_path", type=str, default="resources/test.csv", help="test path")
    parser.add_argument("--cls_label_path", type=str, default="resources/label.json", help="label path")
    parser.add_argument("--output_path", type=str, default="output", help="output path")
    parser.add_argument("--ckpt_path", type=str, required=False, help="checkpoint path")
    return parser.parse_args()