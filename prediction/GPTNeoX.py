from transformers import GPTNeoXJapaneseForCausalLM, GPTNeoXJapaneseTokenizer

model = GPTNeoXJapaneseForCausalLM.from_pretrained("abeja/gpt-neox-japanese-2.7b")
tokenizer = GPTNeoXJapaneseTokenizer.from_pretrained("abeja/gpt-neox-japanese-2.7b")

dir_name = "gpt-neox-japanese-2.7b/"
model.save_pretrained("../model/" + dir_name)

prompt = "そして輝くウルトラ"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

print(gen_text)
