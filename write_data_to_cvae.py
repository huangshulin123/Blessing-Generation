import pandas as pd

def write_data(input_file, output_title_file, output_content_file, is_train=True):
    data = pd.read_csv(input_file)
    if not is_train:
        data = data.groupby(["Scene", "Object"]).apply(lambda x: x.iloc[:20]).reset_index(drop=True)
    title_file_handle = open(output_title_file, "w", encoding="utf-8")
    content_file_handle = open(output_content_file, "w", encoding="utf-8")
    for i, row in data.iterrows():
        title = f"Send this blessing to {row['Object']} for {row['Scene']}:"
        content = row["Content"]
        title_file_handle.write(title + "\n")
        content_file_handle.write(content + "\n")

cvae_base_dir = "CVAE/data/blessing"
write_data("resources/train.csv", f"{cvae_base_dir}/train_title.txt", f"{cvae_base_dir}/train_content.txt", is_train=True)
write_data("resources/test.csv", f"{cvae_base_dir}/val_title.txt", f"{cvae_base_dir}/val_content.txt", is_train=False)
write_data("resources/test.csv", f"{cvae_base_dir}/test_title.txt", f"{cvae_base_dir}/test_content.txt", is_train=False)