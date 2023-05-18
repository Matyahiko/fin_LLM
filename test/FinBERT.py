from transformers import AutoTokenizer, AutoModelForMaskedLM,BertForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("izumi-lab/bert-small-japanese-fin")
model = AutoModelForMaskedLM.from_pretrained("izumi-lab/bert-small-japanese-fin")

#text = "流動資産は、1億円となりました。"
text = "子供にとって、ゲームは害になるのか？"


# prepend [CLS] 【注1】
text = "[CLS]" + text 
# tokenize 
tokens = tokenizer.tokenize(text) 
print(tokens) 
# mask a token 【注2】
masked_idx = 6
tokens[masked_idx] = tokenizer.mask_token 
print(tokens) 
# convert to ids 
token_ids = tokenizer.convert_tokens_to_ids(tokens) 
print(token_ids) 
#exit() 
# convert to tensor 
token_tensor = torch.tensor([token_ids]) 
# get the top 10 predictions of the masked token 
model = model.eval() 
with torch.no_grad(): 
    outputs = model(token_tensor) 
    predictions = outputs[0][0, masked_idx].topk(10) 
for i, index_t in enumerate(predictions.indices): 
    index = index_t.item() 
    token = tokenizer.convert_ids_to_tokens([index])[0] 
    print(i, token)