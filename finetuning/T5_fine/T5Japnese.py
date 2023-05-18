from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from datasets import load_dataset

# データセットのロード（この例ではChatbotデータセットを使用）
dataset = load_dataset("../datasets_scripts/corpus/newsplus_shaped.tsv")

# トークナイザの初期化
tokenizer = T5Tokenizer.from_pretrained("sonoisa/t5-base-japanese")

# T5モデルの初期化
model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")

# データセットの前処理
def preprocess_function(examples):
    inputs = [f'Chat: {chat}' for chat in examples['input_text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)

    # ラベルの設定
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 前処理関数の適用
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 学習設定
training_args = TrainingArguments(
    "test-chatbot", 
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# トレーナーの初期化とモデルの学習
trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["validation"]
)

# モデルのファインチューニング
trainer.train()
