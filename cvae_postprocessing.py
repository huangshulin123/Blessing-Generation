import json
import os
import re
from collections import defaultdict

pattern = r"""================================================== SAMPLE .+? ==================================================

======================================== Outlines  ========================================

(Send this blessing to (.+?) for (.+?):)<\|endoftext\|>

======================================== Story ========================================

(.+?)

======================================== Generated ========================================

(.+)"""

vae_dir = "CVAE/out-vae-gpt2/test"
out_dir = "output-vae-gpt2"
generation_file = "generate-0135060.txt"


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = generation_file.split(".")[0] + ".json"
    lines = open(os.path.join(vae_dir, generation_file), "r", encoding="utf-8").read()
    samples = list(filter(lambda x: x.strip(), lines.split("\n" * 4)))
    # print(len(samples))
    # print(samples[0])
    output = defaultdict(list)
    for sample in samples:
        match = re.match(pattern, sample, flags=re.S | re.M)
        assert match is not None or "Generated (Posterior)" in sample, sample
        if match is None:
            continue
        prompt, object_, scene, reference, candidate = match.groups()
        # print(prompt, scene, object_, reference, candidate)
        output[(scene, object_, prompt)].append((candidate.replace("<|endoftext|>", ""), reference.replace("<|endoftext|>", "")))
    output = [{"scene": key[0], "object": key[1], "prompt": key[2], "candidates": list(map(lambda x: x[0], value)), "references": list(map(lambda x: x[1], value))} for key, value in output.items()]
    json.dump(output, open(os.path.join(out_dir, output_file), "w", encoding="utf-8"), ensure_ascii=False, indent=2)