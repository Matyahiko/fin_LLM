from transformers import AutoTokenizer, AutoModel,T5Tokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese")

model = T5ForConditionalGeneration.from_pretrained("sonoisa/t5-base-japanese")

dir_name = "sonoisa/t5-base-japanese/"
model.save_pretrained("../models/" + dir_name)

input_ids = tokenizer("summarize:被爆による壊滅的な被害、そして見事に復興を遂げ、いま世界に向けて平和の大切さを訴えている広島においてG7首脳が集うこと、この意味は大変大きいと思っています。いま国際的に力による一方的な現状変更が行われるなど不透明な状況にある中で、いま平和を語る、平和へのコミットメントを示すうえで広島ほどふさわしい場所はないと思っています。ぜひこの広島において、後退していると言われている核兵器のない世界を目指すという国際的な機運、これを再び盛り上げる反転させるきっかけを得たいと思っています。そのためにはまずG7において核兵器のない世界を目指すという思いを一致させたうえで世界に向けてそういった思いを訴え、具体的な実践的な取り組みを進めることを議論し進めていかなければならないと思っています。", return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
