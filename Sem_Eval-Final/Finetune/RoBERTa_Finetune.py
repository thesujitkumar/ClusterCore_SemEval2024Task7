import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW
from tqdm import tqdm
class HeadlineDataset(Dataset):
    def __init__(self, news_bodies, headlines, tokenizer, max_length=250):
        self.news_bodies = news_bodies
        self.headlines = headlines
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.news_bodies)

    def __getitem__(self, idx):
        news_body = self.news_bodies[idx]
        headline = self.headlines[idx]

        # Tokenize news body and headline
        inputs = self.tokenizer.encode_plus(news_body, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        labels = self.tokenizer.encode(headline, truncation=True, padding='max_length', max_length=12, return_tensors='pt')

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_gigaword")

# Load and preprocess the dataset
train_df = pd.read_csv("train_headline_gen.csv")  # Assuming your CSV file is named "train_headlines.csv"
# train_df = train_df.head(10)
val_df = pd.read_csv("val_set.csv")
# val_df = val_df.head(10)
# Define train and validation datasets
train_dataset = HeadlineDataset(train_df["news"].tolist(), train_df["headline"].tolist(), tokenizer)
val_dataset = HeadlineDataset(val_df["news"].tolist(), val_df["headline"].tolist(), tokenizer)

# Define train and validation dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model.to(device)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 200
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training loop
    model.train()
    total_loss = 0
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

    # Calculate average loss for epoch
    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

    average_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}")

    # Save the model if validation loss improves
    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        model.save_pretrained("best_model")

print("Training finished.")
