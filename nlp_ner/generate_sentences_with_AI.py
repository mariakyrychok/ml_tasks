import nltk
import pandas as pd
import os
import re
from nltk.tokenize import sent_tokenize
from openai import OpenAI

nltk.download('punkt_tab')

mountains = pd.read_csv('data/top_100_mountain_names.csv')['Mountain Name'].tolist()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def contains_possessive_form(sentence, mountain):
    # check if the mountain name is used in a possessive form
    pattern = rf"{re.escape(mountain)}'s"
    return bool(re.search(pattern, sentence, re.IGNORECASE))

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
            if mountain.lower() in sent.lower() and not contains_possessive_form(sent, mountain):
                generated_data.loc[len(generated_data)] = [sent, mountain]
    return generated_data


dataset = generate_mountain_sentences(mountains)
dataset.to_csv('data/sentences.csv', index=False)
#%%
