from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, random_split
from 

# モデルとトークナイザーを準備
model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")
tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")

# データの読み込み
#data = pd.read_csv("../datasets_scripts/corpus/newsplus_input.tsv", sep="\t")

#input_texts = data["input_text"].tolist()
#target_texts = data["target_text"].tolist()
             
with open("../../datasets_scripts/corpus/newsplus_input.json", "r") as f:  
    data = json.load(f)

input_texts = [item['input_text'] for item in data]
target_texts = [item['target_text'] for item in data]


# データのトークナイズ
inputs = tokenizer(input_texts, truncation=True, padding=True, max_length=512)
targets = tokenizer(target_texts, truncation=True, padding=True, max_length=512)

# データを学習用と評価用に分割
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs['input_ids'], targets['input_ids'], test_size=0.2)

# 学習の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
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
