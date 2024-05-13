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
        ground_truth_headline = self.tokenizer.encode(headline, truncation=True, padding='max_length', max_length=12, return_tensors='pt')

        return {'input_text': input_text.squeeze(), 'ground_truth_headline': ground_truth_headline.squeeze()}


model_name = 'MU-NLPC/CzeGPT-2_headline_generator'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained(model_name)

train_df = pd.read_csv("train_headline_gen.csv")  # Assuming your CSV file is named "news_data.csv"
# train_df = train_df.head(1)

train_news_bodies = train_df["news"].tolist()
train_headlines = train_df["headline"].tolist()

val_df = pd.read_csv("val_set.csv")
# val_df = val_df.head(1)




val_news_bodies = val_df["news"].tolist()
val_headlines = val_df["headline"].tolist()
train_df= train_df.reset_index(drop=True)
train_dataset = NewsDataset(train_news_bodies, train_headlines, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

val_dataset = NewsDataset(val_news_bodies, val_headlines, tokenizer)
# val_df= val_df.reset_index(drop=True)
# val_dataset = NewsDataset(val_df.head())
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,  collate_fn=custom_collate_fn)

# Fine-tune GPT-2 model
# Fine-tune GPT-2 model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# device = "cpu"
model.to(device)
model.train()
learning_rate = 2e-5
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
num_epochs = 200

best_val_loss = float('inf')
best_model_path = "GPT2_4_Headline_Generations.pth"

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    num_batches = len(train_dataloader)

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        input_texts = batch["input_texts"].to(device)
        ground_truth_headlines = batch["ground_truth_headlines"].to(device)

        outputs = model(input_texts, labels=ground_truth_headlines)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"Validation", leave=False):
            # print(batch.keys())
            input_texts = batch["input_texts"].to(device)
            ground_truth_headlines = batch["ground_truth_headlines"].to(device)

            outputs = model(input_texts, labels=ground_truth_headlines)
            val_loss += outputs.loss.item()

    average_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Validation Loss: {average_val_loss:.4f}")

    # Save the best model for future loading
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        model.save_pretrained("GPT_2_Best_model_finetune")
print("Training finished. Best model saved for future loading.")
