#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()

parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--teacher_model', default="facebook/galactica-1.3b", type=str)
parser.add_argument('--dataset', default="covidqa", type=str)
parser.add_argument('--stanza_ent_file', type=str)
parser.add_argument('--num_of_ents', default=10, type=int)
parser.add_argument('--num_of_ctx_per_ent', default=5, type=int)

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

most_common_entities = Counter(stanza_ents).most_common()[:args.num_of_ents]
selected_ents_text_dict = {}
print('Entities to generate contexts for selected...')

generator_model_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
generator_model = AutoModelForCausalLM.from_pretrained(args.teacher_model)
generator = pipeline('text-generation', model = generator_model, tokenizer=generator_model_tokenizer, device=0) 
print('Teacher model loaded...')

for tup in tqdm(most_common_entities):
    entity = tup[0]
    final_string = ''
    
    for i in range(args.num_of_ctx_per_ent):
        set_seed(i)
        final_string = final_string + ' ' + generator(f'{entity}', renormalize_logits=True, do_sample=True, max_length=2048, top_p=0.9, temperature=0.9, use_cache=True)[0]['generated_text']
    
    selected_ents_text_dict[entity] = final_string.strip()

print('Saving mini_corpus...')

pd.DataFrame(zip(selected_ents_text_dict.keys(), selected_ents_text_dict.values()), columns = ['ent', 'text']).to_csv('extended-PT-galactica-MLM-mini_corpus.csv', index=False)