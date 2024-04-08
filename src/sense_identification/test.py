from transformers import AutoTokenizer
import pandas as pd
import os
from itertools import combinations

def find_vocab_idx(word, tokenization):
    if model_checkpoint in ['tiiuae/falcon-7b-instruct', 'garage-bAInd/Platypus2-7B', 'google/gemma-7b-it',
                            'mistralai/Mistral-7B-Instruct-v0.2', 'roberta-base']:
        if word in tokenizer.vocab.keys():
            if tokenizer.vocab[word] in tokenization['input_ids'].tolist()[0]:
                return tokenizer.vocab[word]

        if ('Ġ' + word.lower()) in tokenizer.vocab.keys():
            if tokenizer.vocab['Ġ' + word.lower()] in tokenization['input_ids'].tolist()[0]:
                word = 'Ġ' + word.lower()
                return tokenizer.vocab[word]

        if ('Ġ' + word) in tokenizer.vocab.keys():
            if tokenizer.vocab['Ġ' + word] in tokenization['input_ids'].tolist()[0]:
                word = 'Ġ' + word
                return tokenizer.vocab[word]


model_checkpoint = 'tiiuae/falcon-7b-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

df = pd.read_csv(os.path.abspath(f'../../data/sense_data/sense_data-covidqa.csv'))

for word in df['word'].unique():
    word_indices = df[df['word'] == word].index
    for comb in list(combinations(list(range(word_indices[0], word_indices[-1] + 1)), 2)):
        indexA = comb[0]
        indexB = comb[1]

        print(comb)