#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModel, logging
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import argparse
import torch
import os

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model_name', type=str)
args = parser.parse_args()

model_checkpoint = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

df = pd.read_csv(os.path.abspath(f'../../data/sense_data/{args.dataset}'))

cos = torch.nn.CosineSimilarity()
sim_scores = defaultdict(list)

for word in df['word'].unique():
    word_indices = df[df['word'] == word].index
    for comb in list(combinations(list(range(word_indices[0], word_indices[-1]+1)), 2)):
        indexA = comb[0]
        indexB = comb[1]

        tokenized_inputA = tokenizer(df.iloc[indexA].example, return_tensors='pt') 
        pooler_outputA = model(**tokenized_inputA.to(device)).pooler_output

        tokenized_inputB = tokenizer(df.iloc[indexB].example, return_tensors='pt') 
        pooler_outputB = model(**tokenized_inputB.to(device)).pooler_output

        sim_scores[(word, df.iloc[comb[0]].sense_def, df.iloc[comb[1]].sense_def)].append(cos(pooler_outputA, pooler_outputB).item())

print(f'Model: {model_checkpointl}')

for key, val in sim_scores.items():
    print(key, np.round(torch.mean(torch.Tensor(val)).item(), 2))