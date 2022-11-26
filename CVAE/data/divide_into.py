from collections import defaultdict

import pandas as pd
from tqdm import tqdm

dic_br = defaultdict(list)
df = pd.read_csv("total.csv")
dir_title = "title.txt"
dir_content = "content.txt"
f_title_train = open(f"blessing/train_{dir_title}", "w", encoding="utf-8")
f_title_val = open(f"blessing/val_{dir_title}", "w", encoding="utf-8")
f_title_test = open(f"blessing/test_{dir_title}", "w", encoding="utf-8")
f_content_train = open(f"blessing/train_{dir_content}", "w", encoding="utf-8")
f_content_val = open(f"blessing/val_{dir_content}", "w", encoding="utf-8")
f_content_test = open(f"blessing/test_{dir_content}", "w", encoding="utf-8")
for i in range(0, len(df)):
    fes = df.iloc[i, 0]
    target = df.iloc[i, 1]
    content = df.iloc[i, 2]
    title = f"Send this blessing to {target} for {fes}:"
    dic_br[title].append(content)
for key, value in tqdm(dic_br.items()):
    number = int(len(value) / 10)
    for i in range(0, 8 * number):
        f_title_train.write(key + "\n")
        f_content_train.write(value[i].replace("\n", " ") + "\n")
    for i in range(8 * number, 9 * number):
        f_title_val.write(key + "\n")
        f_content_val.write(value[i].replace("\n", " ") + "\n")
    for i in range(9 * number, min(len(value), 9 * number + 20)):
        f_title_test.write(key + "\n")
        f_content_test.write(value[i].replace("\n", " ") + "\n")
f_title_train.close()
f_title_val.close()
f_title_test.close()
f_content_train.close()
f_content_val.close()
f_content_test.close()
