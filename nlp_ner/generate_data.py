import csv

import nltk
import pandas as pd
import os
from nltk.tokenize import sent_tokenize
from openai import OpenAI

nltk.download('punkt_tab')

mountains = pd.read_csv('datasets/top_100_mountain_names.csv')['Mountain Name'].tolist()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def generate_mountain_sentences(mountain_list, num_sentences=2):
    generated_data = pd.DataFrame(columns=['sentence', 'mountain'])
    for mountain in mountain_list:
        prompt = f'Write {num_sentences} sentences mentioning {mountain} name.'
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=300,
        )
        text = response.choices[0].message.content.strip()
        sentences = sent_tokenize(text)
        for sent in sentences:
            if mountain.lower() in sent.lower():
                generated_data.loc[len(generated_data)] = [sent, mountain]
    return generated_data


dataset = generate_mountain_sentences(mountains)
dataset.to_csv('datasets/sentences.csv', index=False)

#%%
