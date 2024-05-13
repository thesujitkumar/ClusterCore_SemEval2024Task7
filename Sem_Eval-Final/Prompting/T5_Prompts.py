import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_headline(news_body, prompt, model_name_or_path="t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    # Preprocess the inputs and convert them to tensors
    input_text = prompt + news_body
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate headline
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=24, num_beams=4, early_stopping=True)

    # Decode the generated headline and return it
    generated_headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_headline

# Read input CSV file
input_file_path = "test_set.csv"
output_file_path = "T5_head_prompt.csv"

input_data = pd.read_csv(input_file_path)
input_data = input_data.head(100)
# Custom prompt for generating headlines
custom_prompt = "I want you to generate a short headline for a given news article. The headline should be concise and samll but represent the content of news body. The headline may contain a number which could be obtained by perforimg simple arithematic operations like addition, subtraction, division and multiplication or obtained by copying the same valid number from the news article if required to summarize the article.: "

# Generate headlines for each news article
generated_headlines = []
for index, row in input_data.iterrows():
    news_body = row["news"]
    headline = generate_headline(news_body, custom_prompt)
    print(headline)
    generated_headlines.append(headline)

# Add generated headlines to the DataFrame
input_data["generated_headline"] = generated_headlines

# Write the DataFrame with generated headlines to output CSV file
input_data.to_csv(output_file_path, index=False)

print("Generated headlines saved to:", output_file_path)
