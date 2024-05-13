import os
os.environ["HF_HOME"] = "/data/btp_data/Siddharth/huggingface_cache/"

from datasets import load_dataset, load_from_disk
import numpy as np
import nltk
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import torch
from datasets import load_metric
from ctransformers import AutoModelForCausalLM
import configparser

config = configparser.ConfigParser()
config.read('inference_config.ini')

dataset = pd.read_csv(config['dataset']['data_path_test'])
news_texts = dataset['news']
news_list = list(news_texts)

model_id = config['model']['model_id']

# check ctransformers doc for more configs
config_llm = {
          'max_new_tokens': int(config['model']['max_new_tokens']),
          'repetition_penalty': float(config['model']['repetition_penalty']),
          'temperature': float(config['model']['temperature']),
          'stream': bool(config['model']['stream']),
          'context_length': int(config['model']['context_length'])
          }

llm = AutoModelForCausalLM.from_pretrained(
      model_id,
      model_type="llama",
      #lib='avx2', for cpu use
      gpu_layers=130, #110 for 7b, 130 for 13b
      **config_llm
      )

prompt = '''<s>[INST] {{ I want you to generate a headline for a given news article. 
The headline should be concise.
The headline may contain a number if required to summarize the article.
Here is an example:
News article:
(Jan 10, 2018  7:10 PM) Authorities now say 17 people have died in Southern California mudslides and another 13 are missing, the AP reports. The death toll rose Wednesday as searchers pulled two more bodies from the inundated area in the Santa Barbara County enclave of Montecito. Flash floods there on Tuesday swept immense amounts of mud, water, and debris down from foothills that were stripped of brush by the recent Thomas wildfire. Authorities say at least 100 homes have been destroyed. Hundreds of firefighters and others are hunting through the mud and wreckage. Three people were rescued Wednesday and authorities say about 75%% of the devastated area has been searched.
Headline:
Death Toll From California Mudslide Rises to 17

Now solve it for this example:
Article:
%s
}} [/INST]'''

headlines = []
for counter, news in tqdm(list(enumerate(news_list))):
    headlines.append(llm(prompt%news, stream=False))
    if counter%10 ==0:
        result = pd.DataFrame(headlines)
        result.to_csv(config['dataset']['output_path_test'], index = False)
    print(headlines[-1])