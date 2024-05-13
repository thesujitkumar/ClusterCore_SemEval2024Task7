# !pip install datasets
import sacrebleu
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from tqdm import tqdm
import sacrebleu
import os


df=pd.read_csv("train_headline_gen.csv")
print(" the number of sample in training datatset", len(df))
# df=df.head(10)


# Define a class for the summarization dataset
class SummarizationDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data.iloc[idx]["news"]
        summary = self.data.iloc[idx]["headline"]

        # Tokenize the data
        encoding = self.tokenizer(
            document,
            summary,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": self.tokenizer(summary, return_tensors="pt", max_length=250, truncation=True, padding="max_length")["input_ids"].squeeze(),
        }

# Create train, validation, and test datasets
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")
train_dataset = SummarizationDataset(df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# val_df = pd.DataFrame({'dialogue': [example['dialogue'] for example in dataset['validation']],
#                        'summary': [example['summary'] for example in dataset['validation']]})


val_df = pd.read_csv("val_set.csv")
print(" the number of sample in validation datatset", len(val_df))
# val_df = val_df.head(10)



val_dataset = SummarizationDataset(val_df, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# test_df = pd.DataFrame({'dialogue': [example['dialogue'] for example in dataset['test']],
#                         'summary': [example['summary'] for example in dataset['test']]})
# test_df = test_df.head(10)

test_df = pd.read_csv("test_set.csv")
print(" the number of sample in training datatset", len(test_df))
# test_df = test_df.head(10)


test_dataset = SummarizationDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Initialize the model and optimizer
model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = "cpu"
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Set the number of training epochs
num_epochs = 100

# Initialize best BLEU score
best_bleu = 0.0
best_model_state_dict = None

# Training loop
for epoch in range(num_epochs):
    print(f"Training Epoch {epoch + 1}/{num_epochs}")

    # Set the model to training mode
    model.train()

    # Training loop for each batch
    for batch in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    # Set the model to evaluation mode
    model.eval()

    # Initialize lists to store reference summaries and model-generated summaries
    references = []
    hypotheses = []

    # Validation loop
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validating...", leave=False):
            input_ids = val_batch["input_ids"].to(device)
            attention_mask = val_batch["attention_mask"].to(device)
            labels = val_batch["labels"].to(device)

            # Generate summaries using the model
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=12, num_beams=2, length_penalty=2.0, early_stopping=True)
            generated_summaries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            references.append([tokenizer.decode(label, skip_special_tokens=True) for label in labels])
            hypotheses.append(generated_summaries)

    # Flatten the list of references
    references = [ref for sublist in references for ref in sublist]

    # Compute BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, BLEU Score: {bleu.score}")
    with open('bleu_scores.txt', 'a') as bleu_file:
        # Write final BLEU score to file
        bleu_file.write(f" epoch {epoch},  BLEU Score on validation Set: {bleu.score}\n")
    # Update best BLEU score and save the best model
    if bleu.score > best_bleu:
        best_bleu = bleu.score
        best_model_state_dict = model.state_dict()

        output_dir = "Best_so_far_fintune_T5_4_Sum"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)



output_dir = "Best_fintune_T5_4_Sum"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
model.load_state_dict(best_model_state_dict)

# Set the model to evaluation mode
model.eval()

# Initialize lists for test references and hypotheses
test_references = []
test_hypotheses = []

#test loop
with torch.no_grad():
    for test_batch in tqdm(test_loader, desc="Testing...", leave=False):
        input_ids = test_batch["input_ids"].to(device)
        attention_mask = test_batch["attention_mask"].to(device)
        labels = test_batch["labels"].to(device)

        # Generate summaries using the best model
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=12, num_beams=2, length_penalty=2.0, early_stopping=True)
        generated_summaries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        test_references.append([tokenizer.decode(label, skip_special_tokens=True) for label in labels])
        test_hypotheses.append(generated_summaries)

# Flatten the list of references
test_references = [ref for sublist in test_references for ref in sublist]

# Compute BLEU score on the entire test set
test_bleu = sacrebleu.corpus_bleu(test_hypotheses, [test_references])
print(f"Final BLEU Score on Test Set: {test_bleu.score}")
with open('bleu_scores.txt', 'a') as bleu_file:
    # Write final BLEU score to file
    bleu_file.write(f"Final BLEU Score on Test Set: {test_bleu.score}\n")
