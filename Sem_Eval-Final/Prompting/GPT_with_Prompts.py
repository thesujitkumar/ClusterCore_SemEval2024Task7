import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_headline(news_body, prompt, max_length=12, temperature=0.7, top_k=0, top_p=0.9):
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M", ignore_mismatched_sizes=True)
    model = GPT2LMHeadModel.from_pretrained("EleutherAI/gpt-neo-125M",ignore_mismatched_sizes=True)

    input_text = news_body + " " + prompt
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    headline = tokenizer.decode(output[0], skip_special_tokens=True)
    return headline

def generate_headlines_from_csv(input_csv, output_csv, prompt):
    df = pd.read_csv(input_csv)

    generated_headlines = []
    for index, row in df.iterrows():
        news_body = row['news']
        generated_headline = generate_headline(news_body, prompt)
        generated_headlines.append(generated_headline)
        print(generated_headline)

    df['Generated Headline'] = generated_headlines
    df.to_csv(output_csv, index=False)

# Example usage
input_csv = "test_set.csv"
output_csv = "GPT_Model_Prompt_headline.csv"
prompt = "I want you to generate a headline for a given news article. The headline should be concise and samll but represent news body. The headline may contain a number which could be obtained by perforimg simple arithematic operations like addition, subtraction, division and multiplication or obtained by copying the same valid number from the news article if required to summarize the article."
generate_headlines_from_csv(input_csv, output_csv, prompt)
print("Generated headlines saved to:", output_csv)
