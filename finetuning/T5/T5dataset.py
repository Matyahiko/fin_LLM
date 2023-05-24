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

class TsvDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, input_max_len=512, target_max_len=512):
        #self.file_path = os.path.join(data_dir, type_path)
        self.file_path = data_dir
        self.input_max_len = input_max_len
        self.target_max_len = target_max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
  
    def __len__(self):
        return len(self.inputs)
  
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        source_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": source_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _make_record(self, title, body):
        # ニュースタイトル生成タスク用の入出力形式に変換する。
        input = f"{title}"
        target = f"{body}"
        return input, target
  
    def _build(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("\t")
                #assert len(line) == 3
                assert len(line[0]) > 0
                assert len(line[1]) > 0
                #assert len(line[2]) > 0

                title = line[0]
                body = line[1]
                #genre_id = line[2]
                


                input, target = self._make_record(title, body)

                tokenized_inputs = self.tokenizer.batch_encode_plus(
                    [input], max_length=self.input_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )
                #print(tokenized_inputs)

                tokenized_targets = self.tokenizer.batch_encode_plus(
                    [target], max_length=self.target_max_len, truncation=True, 
                    padding="max_length", return_tensors="pt"
                )
                #print(tokenized_targets)
                self.inputs.append(tokenized_inputs)
                self.targets.append(tokenized_targets)