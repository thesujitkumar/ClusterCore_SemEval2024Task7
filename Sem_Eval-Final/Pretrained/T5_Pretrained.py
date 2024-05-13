import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
# import sacrebleu
import torch

df= pd.read_csv("test_sem.csv")
df=df.head(10)
news_texts = df['news']
news_list = list(news_texts)

generated_summary = []

tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

sum=[]
gen_head=[]
for a in news_list:
    input_ids = tokenizer(a, return_tensors="pt", max_length=250, padding=True,truncation=True).to(device)
    summary_ids = model.generate(input_ids['input_ids'], max_length=15, min_length = 10, num_beams=4, length_penalty=2.0, early_stopping=True)
    generated_headlines = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    print(generated_headlines)
    gen_head.append(generated_headlines)
    sum.extend(generated_headlines)
# reference = df['summary']
# ref_list = list(reference)

# df = pd.DataFrame(gen_head)
# df.to_csv("Test_sem_T5_with_FineTun.csv", index=False)




# for i in sum:
#     print(type(i))

# print( " the number of sample in generated summary is", len(sum))
# print( " the number of sample in generareference summary is", len(ref_list))
#
# bleu = sacrebleu.corpus_bleu(sum, [ref_list])
# print("BLEU Score:", bleu.score)
#
# bleu = sacrebleu.corpus_bleu(sum, [ref_list])
# normalized_bleu = bleu.score / 100.0
# print("Normalized BLEU Score:", normalized_bleu)
