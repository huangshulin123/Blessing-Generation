# Blessing-Generation

*Dataset and Models for GEM 2022 Workshop*

*"Towards Attribute-Entangled Controllable Text Generation: A Pilot Study of Blessing Generation".*

## Data

**The full processed dataset is placed in `resources/all_data.csv`.**

## GPT2, T5 Models and Evaluation

### Requirements

- torch == 1.10.0
- transformers == 4.14.1
- pytorch-lightning == 1.6.2
- scikit-learn == 1.0.2
- nltk == 3.6.3

### Prepare Data

The overall dataset (csv file) should be divide into two files `train.csv` and `test.csv`. Put them in the `resources` directory.

### Train

Run the following command to train a blessing generation model:

```shell
python -u main.py \
    --output_path output \
    --model_name gpt2 \
    --max_length 200
```

The model name `gpt2` can be replaced with `t5-base` for training T5 model.

After each training epoch, the checkpoint will be saved in the `output` directory.

### Test

Run the following command to test a model:

```shell
python -u main.py \
    --output_path output \
    --model_name gpt2 \
    --max_length 200 \
    --mode test \
    --ckpt_path <ckpt_file> # <ckpt_file> should be replaced with the specific filename of the checkpoint
```

The generated blessing sentences will also be output in the `output` directory in `json` format.

### Evaluation

Modify some variables related to the directory and filename in `metrics.py` and execute it to calculate the metrics.

The metrics are saved in some csv files in the same directory as the prediction file, i.e., `output`.

## GPT2 + CVAE and GPT2 + Adv.

The code is directly borrowed from [TransformerCVAE](https://github.com/fangleai/TransformerCVAE).

### Prepare Data

Execute `write_data_to_cvae.py` to convert the csv files `train.csv` and `test.csv` mentioned in the last section to the input data for CVAE and adversarial architecture.

### Train and Test

Run the following commands for training the two models.

```shell
cd CVAE
python train_adv.py # For training GPT2 + Adv.
python train_cvae.py # For training GPT2 + CVAE
```

During the training process, a checkpoint and a text file containing prediction result in the test dataset will be output every few training rounds.

### Evaluation

Modify and execute `vae_postprocessing.py` to convert the prediction results to a `json` file for calculating the metrics.

Follow the `Evaluation` part of the last section to obtain the all metrics.
