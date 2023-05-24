import pandas as pd
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
import pytorch_lightning as pl

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
    Trainer
)

from T5dataset import TsvDataset

# 乱数シードの設定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(0)

# GPU利用有無
USE_GPU = torch.cuda.is_available()

# 各種ハイパーパラメータ
args_dict = dict(
    PRETRAINED_MODEL_NAME = "sonoisa/t5-base-japanese"
    data_dir="../../dataset/2ch_dataset/corpus/newsplus_input.tsv",  # データセットのディレクトリ
    model_name_or_path=PRETRAINED_MODEL_NAME,
    tokenizer_name_or_path=PRETRAINED_MODEL_NAME,

    learning_rate=3e-4,
    betas=(0.9, 0.999),
    adam_epsilon=1e-8,
    weight_decay=0.0,
    warmup_steps=0,
    gradient_accumulation_steps=1,

    # max_input_length=512,
    # max_target_length=4,
    # train_batch_size=8,
    # eval_batch_size=8,
    # num_train_epochs=4,

    n_gpu=1 if USE_GPU else 0,
    early_stop_callback=False,
    fp_16=False,
    # opt_level='O1',
    max_grad_norm=1.0,
    seed=42,
)



         
# トークナイザー（SentencePiece）モデルの読み込み
tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME, is_fast=True)

# テストデータセットの読み込み
train_dataset = TsvDataset(tokenizer, args_dict["data_dir"], "test.tsv", 
                           input_max_len=512, target_max_len=512)

for data in train_dataset:
    print("A. 入力データの元になる文字列")
    print(tokenizer.decode(data["source_ids"]))
    print()
    print("B. 入力データ（Aの文字列がトークナイズされたトークンID列）")
    print(data["source_ids"])
    print()
    print("C. 出力データの元になる文字列")
    print(tokenizer.decode(data["target_ids"]))
    print()
    print("D. 出力データ（Cの文字列がトークナイズされたトークンID列）")
    print(data["target_ids"])
    break

"""
# 学習の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,B
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainerの準備と学習の開始
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=val_inputs,
)

trainer.train()
"""