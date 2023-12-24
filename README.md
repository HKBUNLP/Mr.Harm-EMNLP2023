# Mr.Harm
Official PyTorch implementation for our EMNLP23 (Findings) paper: **Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models**.

[[`paper`](https://aclanthology.org/2023.findings-emnlp.611/)]


## Install

```bash
conda create -n meme python=3.8
conda activate meme
pip install -r requirements.txt
```

## Data

Please refer to [Data](https://github.com/HKBUNLP/Mr.Harm-EMNLP2023/tree/main/Data).

## Training
- Learn from LLMs
```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=32 batch_size=32 \
    clip32_base224 text_t5_base image_size=224 vit_randaug mode="rationale" \
    log_dir=$LOG precision=32 max_epoch=10 learning_rate=5e-5
```

- Learn from Labels
```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/save/ckpts/name"

rm -rf $LOG
mkdir $LOG

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=32 batch_size=32 \
    clip32_base224 text_t5_base image_size=224 vit_randaug mode="label" \
    log_dir=$LOG precision=32 max_epoch=30 learning_rate=5e-5 \
    load_path="/path/to/distill_LLMs.ckpt"
```

## Inference

```bash
export DATA="/path/to/data/folder"
export LOG="/path/to/log/folder"

CUDA_VISIBLE_DEVICES=0 python run.py with data_root=$DATA \
    num_gpus=1 num_nodes=1 task_train per_gpu_batchsize=1 batch_size=1 \
    clip32_base224 text_t5_base image_size=224 vit_randaug \
    log_dir=$LOG precision=32 test_only=True \
    load_path="/path/to/label_learn.ckpt" \
    out_path="/path/to/save/label_pred.json"
```
Then, you can use the `/path/to/save/label_pred.json` and the gold labels to get the scores.

## Citation

```
@inproceedings{lin-etal-2023-beneath,
    title = "Beneath the Surface: Unveiling Harmful Memes with Multimodal Reasoning Distilled from Large Language Models",
    author = "Lin, Hongzhan  and
      Luo, Ziyang  and
      Ma, Jing  and
      Chen, Long",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.611",
    doi = "10.18653/v1/2023.findings-emnlp.611",
    pages = "9114--9128",
}
```

## Acknowledgements

The code is based on [ViLT](https://github.com/dandelin/ViLT) and [METER](https://github.com/zdou0830/METER/tree/main).
