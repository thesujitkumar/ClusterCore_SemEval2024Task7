import pandas as pd
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW

def custom_collate_fn(batch):
    # Get the lengths of sequences in the batch
    lengths = [len(item['input_text']) for item in batch]

    # Find the maximum length
    max_length = max(lengths)

    # Pad sequences to the maximum length
    padded_input_texts = []
    padded_ground_truth_headlines = []
    for item in batch:
        input_text = item['input_text']
        ground_truth_headline = item['ground_truth_headline']

        # Pad input text
        padding_length = max_length - len(input_text)
        padding_tensor = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
        padded_input_text = torch.cat([input_text, padding_tensor])
        padded_input_texts.append(padded_input_text)

        # Pad ground truth headline (if necessary)
        padding_length = max_length - len(ground_truth_headline)
        padding_tensor = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
        padded_ground_truth_headline = torch.cat([ground_truth_headline, padding_tensor])
        padded_ground_truth_headlines.append(padded_ground_truth_headline)

    return {
        'input_texts': torch.stack(padded_input_texts),
        'ground_truth_headlines': torch.stack(padded_ground_truth_headlines)
    }

class NewsDataset(Dataset):
    def __init__(self, news_bodies, headlines, tokenizer):
        self.news_bodies = news_bodies
        self.headlines = headlines
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.news_bodies)

    def __getitem__(self, idx):
        news_body = self.news_bodies[idx]
        headline = self.headlines[idx]

        # Tokenize news body and headline
        input_text = self.tokenizer.encode(news_body, truncation=True, padding='max_length', max_length=250, return_tensors='pt')
        ground_truth_headline = self.tokenizer.encode(headline, truncation=True, padding='max_length', max_length=15, return_tensors='pt')

        return {'input_text': input_text.squeeze(), 'ground_truth_headline': ground_truth_headline.squeeze()}


test_df = pd.read_csv("test_sem.csv")  # Assuming your CSV file is named "news_data.csv"
# test_df = test_df.head(5)

test_news_bodies = test_df["news"].tolist()
test_headlines = test_df["headline"].tolist()

model_name = 'MU-NLPC/CzeGPT-2_headline_generator'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model = GPT2LMHeadModel.from_pretrained("GPT_2_Best_model_finetune")
model = GPT2LMHeadModel.from_pretrained("GPT_2_Best_model_finetune")


#train_df= train_df.reset_index(drop=True)
test_dataset = NewsDataset(test_news_bodies, test_headlines, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device="cpu"
model.to(device)

gen_headline=[]
generated_headlines = []
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        input_texts = batch["input_texts"]
        output = model.generate(input_texts.to(model.device), max_length=15, num_return_sequences=1)
        generated_headline = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_headlines.append(generated_headline)
        print(generated_headline)
        gen_headline.append(generated_headline)

# Save generated headlines to CSV file
output_df = pd.DataFrame(gen_headline)
output_df.to_csv("GPT_2_test_sem_Finetune.csv", index=False)
