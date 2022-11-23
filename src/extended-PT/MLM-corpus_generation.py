#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer
from collections import defaultdict
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import pandas as pd
import wikipedia
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_checkpoint', default="bert-base-uncased", type=str)
parser.add_argument('--dataset', default="covidqa", type=str)
parser.add_argument('--stanza_ent_file', type=str)
parser.add_argument('--num_of_ents', default=10, type=int)
parser.add_argument('--num_of_ctx_per_ent', default=1, type=int)

args = parser.parse_args()

stanza_ents = pickle.load(open(args.stanza_ent_file, 'rb'))

tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model_vocab = list(tokenizer.vocab.keys())

# We only want to deal with those entities that are not in the model vocabulary otherwise we would get multiple embeddings for the same term.
ent_in_model_vocab = []
for ent in tqdm(stanza_ents):
    if ent in model_vocab:
        ent_in_model_vocab.append(ent)

for ent in ent_in_model_vocab:
    stanza_ents.remove(ent)

s = Counter(stanza_ents)
sorted_ent_counts = dict(sorted(s.items(), key=lambda item: item[1], reverse=True))

no_of_ents_to_select = args.num_of_ents
no_of_results_per_entity = args.num_of_ctx_per_ent

selected_ents_text_dict = defaultdict(list)

for ent in tqdm(sorted_ent_counts.keys()):
    if len(selected_ents_text_dict) == no_of_ents_to_select:
        break
    
    search_results = wikipedia.search(str(ent), results=no_of_results_per_entity)
    
    for res in search_results:
        try:
            selected_ents_text_dict[ent].append(wikipedia.page(res, auto_suggest=False).content)          
        except:
            continue
    
    if len(selected_ents_text_dict[ent]) != no_of_results_per_entity:
        selected_ents_text_dict.pop(ent)

print(f'Entities that were selected: {selected_ents_text_dict.keys()}') 

print('Saving mini_corpus...')

final_ents = [val for val in selected_ents_text_dict.keys() for _ in range(no_of_results_per_entity)]
texts = [item for sublist in selected_ents_text_dict.values() for item in sublist]

pd.DataFrame(zip(final_ents, texts), columns = ['ent', 'text']).to_csv(f'{args.dataset}-mini_corpus.csv', index=False)