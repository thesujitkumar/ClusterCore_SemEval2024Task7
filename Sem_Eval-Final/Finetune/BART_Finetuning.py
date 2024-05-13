import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, AdamW
import sacrebleu
import pandas as pd
from tqdm import tqdm
import os
# Define your dataset class (implement the __len__ and __getitem__ methods)
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=  250, max_target_length=150):
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        source_text = self.data.iloc[idx]["news"]
        target_text = self.data.iloc[idx]["headline"]

        encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"
model.to(device)
# Load your training and validation datasets using SummarizationDataset
# Replace 'your_train_dataset.csv' and 'your_val_dataset.csv' with your dataset filenames
train_data = pd.read_csv('train_headline_gen.csv')
train_data = train_data.head(10)
val_data = pd.read_csv('val_set.csv')
val_data = val_data.head(10)

test_data = pd.read_csv('test_set.csv')
test_data = test_data.head(10)

train_dataset = SummarizationDataset(train_data, tokenizer)
val_dataset = SummarizationDataset(val_data, tokenizer)
test_dataset = SummarizationDataset(test_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 100
best_bleu = 0.0
best_model_state_dict = None
# Training loop
for epoch in range(num_epochs):
    print(f"Training Epoch {epoch + 1}/{num_epochs}")
    model.train()

    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        # print(loss)

        loss.backward()
        optimizer.step()
        # print("I am done with one batch")

    # Validation with BLEU score computation
    # print("Begin with validations")
    model.eval()
    references = []  # List to store reference summaries
    hypotheses = []  # List to store model-generated summaries

    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validating...", leave=False):
            input_ids = val_batch["input_ids"].to(device)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)

            # Generate summaries using the model
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=12, num_beams=2, length_penalty=2.0, early_stopping=True)
            generated_summaries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Ensure that the generated_summaries contain only valid tokens
            # valid_generated_summaries = [token for token in generated_summaries.split() if token.isdigit()]

            references.append([tokenizer.decode(label, skip_special_tokens=True) for label in labels])
            hypotheses.append(generated_summaries)
    references = [ref for sublist in references for ref in sublist]

    # Compute BLEU score
    # print(hypotheses)
    # print(references)
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, BLEU Score: {bleu.score}")
    with open('bleu_scores.txt', 'a') as bleu_file:
        # Write final BLEU score to file
        bleu_file.write(f" epoch {epoch},  BLEU Score on validation Set: {bleu.score}\n")
    # Update best BLEU score and save the best model
    if bleu.score > best_bleu:
        best_bleu = bleu.score
        best_model_state_dict = model.state_dict()
        best_model = model

        output_dir = "Best_so_far_BART_Sum"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)

# Save the fine-tuned model
output_directory = 'fine_tuned_bart_model'
model.save_pretrained(output_directory)
tokenizer.save_pretrained(output_directory)

output_dir = "Best_BART_Sum"
os.makedirs(output_dir, exist_ok=True)
best_model.save_pretrained(output_dir)

model.load_state_dict(best_model_state_dict)



references = []  # List to store reference summaries
hypotheses = []  # List to store model-generated summaries
with torch.no_grad():
    for test_batch in tqdm(test_loader, desc="Testing...", leave=False):
        input_ids = test_batch["input_ids"].to(device)
        attention_mask = test_batch["attention_mask"].to(device)
        labels = test_batch["labels"].to(device)

        # Generate summaries using the model
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=12, num_beams=2, length_penalty=2.0, early_stopping=True)
        generated_summaries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Ensure that the generated_summaries contain only valid tokens
        # valid_generated_summaries = [token for token in generated_summaries.split() if token.isdigit()]

        references.append([tokenizer.decode(label, skip_special_tokens=True) for label in labels])
        hypotheses.append(generated_summaries)
references = [ref for sublist in references for ref in sublist]

# Compute BLEU score
# print(hypotheses)
bleu = sacrebleu.corpus_bleu(hypotheses, [references])
print(" Test set BLEU Score:", bleu.score)
