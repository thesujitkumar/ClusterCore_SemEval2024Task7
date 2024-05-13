import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

class HeadlineDataset(Dataset):
    def __init__(self, news_bodies, tokenizer, max_length=512):
        self.news_bodies = news_bodies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.news_bodies)

    def __getitem__(self, idx):
        news_body = self.news_bodies[idx]

        inputs = self.tokenizer(news_body, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'news_body': news_body
        }

# Load the trained model
tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword")
model = AutoModelForSeq2SeqLM.from_pretrained("fine_tuned_roberta_gigaword_headline_generator")

# Load test data
test_df = pd.read_csv("test_sem.csv")
# test_df= test_df.head(10)
test_dataset = HeadlineDataset(test_df["news"].tolist(), tokenizer)

# Define data loader
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model.to(device)

# Generate headlines over the test set
predictions = []
with torch.no_grad():
    model.eval()
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        news_bodies = batch['news_body']

        generated_headlines = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64, num_beams=4, early_stopping=True)
        decoded_headlines = tokenizer.batch_decode(generated_headlines, skip_special_tokens=True)

        for news_body, headline in zip(news_bodies, decoded_headlines):
            predictions.append({'news_body': news_body, 'headline': headline})

# Save predictions to CSV file
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv("RoBERTa_Finetune_test_Sem.csv", index=False)
