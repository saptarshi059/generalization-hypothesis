#!/usr/bin/env python
# coding: utf-8

from transformers import AutoTokenizer, AutoModel, logging
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import os

logging.set_verbosity(50)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--pooler', type=str2bool)
args = parser.parse_args()

model_checkpoint = args.model_name

if model_checkpoint != 'sensebert-base-uncased':
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if model_checkpoint in ['tiiuae/falcon-7b-instruct', 'garage-bAInd/Platypus2-7B', 'google/gemma-7b-it',
                            'mistralai/Mistral-7B-Instruct-v0.2']:
        model = AutoModel.from_pretrained(model_checkpoint, device_map='auto', torch_dtype=torch.float16,
                                          attn_implementation="flash_attention_2")
    else:
        model = AutoModel.from_pretrained(model_checkpoint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

df = pd.read_csv(os.path.abspath(f'../../data/sense_data/{args.dataset}'))

sim_scores = defaultdict(list)

# Using pooler output
if args.pooler == True:
    cos = torch.nn.CosineSimilarity()
    for word in df['word'].unique():
        word_indices = df[df['word'] == word].index
        for comb in list(combinations(list(range(word_indices[0], word_indices[-1] + 1)), 2)):
            indexA = comb[0]
            indexB = comb[1]

            tokenized_inputA = tokenizer(df.iloc[indexA].example, return_tensors='pt')
            tokenized_inputB = tokenizer(df.iloc[indexB].example, return_tensors='pt')

            with torch.no_grad():
                pooler_outputA = model(**tokenized_inputA.to(device)).pooler_output
                pooler_outputB = model(**tokenized_inputB.to(device)).pooler_output

            sim_scores[(word, df.iloc[comb[0]].sense_def, df.iloc[comb[1]].sense_def)].append(
                cos(pooler_outputA, pooler_outputB).item())
# Using contextualized entity output
else:
    if model_checkpoint != 'sensebert-base-uncased':
        cos = torch.nn.CosineSimilarity(dim=0)
        sim_scores = defaultdict(list)

        def find_vocab_idx(word, tokenization):
            if model_checkpoint in ['tiiuae/falcon-7b-instruct', 'garage-bAInd/Platypus2-7B', 'google/gemma-7b-it',
                                    'mistralai/Mistral-7B-Instruct-v0.2', 'roberta-base']:

                # For Falcon & RoBERTA
                if ('Ġ' + word.lower()) in tokenizer.vocab.keys():
                    if tokenizer.vocab['Ġ' + word.lower()] in tokenization['input_ids'].tolist()[0]:
                        word = 'Ġ' + word.lower()

                elif ('Ġ' + word) in tokenizer.vocab.keys():
                    if tokenizer.vocab['Ġ' + word] in tokenization['input_ids'].tolist()[0]:
                        word = 'Ġ' + word

                # For Platypus/Gemma/Mistral
                elif ('▁' + word.lower()) in tokenizer.vocab.keys():
                    if (tokenizer.vocab['▁' + word.lower()]) in tokenization['input_ids'].tolist()[0]:
                        word = '▁' + word.lower()

                elif ('▁' + word) in tokenizer.vocab.keys():
                    if tokenizer.vocab['▁' + word] in tokenization['input_ids'].tolist()[0]:
                        word = '▁' + word

                return tokenizer.vocab[word]

            else:  # For BERT
                if word in tokenizer.vocab.keys():
                    if tokenizer.vocab[word] in tokenization['input_ids'].tolist()[0]:
                        return tokenizer.vocab[word]

                if word.lower() in tokenizer.vocab.keys():
                    if tokenizer.vocab[word.lower()] in tokenization['input_ids'].tolist()[0]:
                        return tokenizer.vocab[word.lower()]

        for word in tqdm(df['word'].unique()):
            word_indices = df[df['word'] == word].index
            for comb in tqdm(list(combinations(list(range(word_indices[0], word_indices[-1] + 1)), 2))):
                indexA = comb[0]
                indexB = comb[1]

                tokenized_inputA = tokenizer(df.iloc[indexA].example, return_tensors='pt')
                tokenized_inputB = tokenizer(df.iloc[indexB].example, return_tensors='pt')

                with torch.no_grad():
                    contextualized_embeddingsA = model(**tokenized_inputA.to(device)).last_hidden_state
                    contextualized_embeddingsB = model(**tokenized_inputB.to(device)).last_hidden_state

                wordA_vocab_idx = find_vocab_idx(df.iloc[indexA].word, tokenized_inputA)
                wordB_vocab_idx = find_vocab_idx(df.iloc[indexB].word, tokenized_inputB)

                entity_embeddingA = contextualized_embeddingsA[0][
                    tokenized_inputA['input_ids'].tolist()[0].index(wordA_vocab_idx)]
                entity_embeddingB = contextualized_embeddingsB[0][
                    tokenized_inputB['input_ids'].tolist()[0].index(wordB_vocab_idx)]

                sim_scores[(word, df.iloc[indexA].sense_def, df.iloc[indexB].sense_def)].append( \
                    cos(entity_embeddingA, entity_embeddingB).item())

                print(sim_scores)

    else:
        import sys
        import tensorflow as tf

        sys.path.append('sense-bert')
        from sensebert import SenseBert

        with tf.Session() as session:
            cos = torch.nn.CosineSimilarity(dim=0)
            sim_scores = defaultdict(list)
            model = SenseBert("sensebert-base-uncased", session=session)  # or sensebert-large-uncased
            tokenizer = model.tokenizer


            def find_vocab_idx(word, input_ids):
                if word in tokenizer.vocab.keys():
                    if tokenizer.vocab[word] in input_ids[0]:
                        return tokenizer.vocab[word]

                if word.lower() in tokenizer.vocab.keys():
                    if tokenizer.vocab[word.lower()] in input_ids[0]:
                        return tokenizer.vocab[word.lower()]


            for word in tqdm(df['word'].unique()):
                word_indices = df[df['word'] == word].index
                for comb in tqdm(list(combinations(list(range(word_indices[0], word_indices[-1] + 1)), 2))):
                    indexA = comb[0]
                    indexB = comb[1]

                    input_idsA, input_maskA = model.tokenize(df.iloc[indexA].example)
                    contextualized_embeddingsA, _, _ = model.run(input_idsA, input_maskA)

                    input_idsB, input_maskB = model.tokenize(df.iloc[indexB].example)
                    contextualized_embeddingsB, _, _ = model.run(input_idsB, input_maskB)

                    wordA_vocab_idx = find_vocab_idx(df.iloc[indexA].word, input_idsA)
                    wordB_vocab_idx = find_vocab_idx(df.iloc[indexB].word, input_idsB)

                    entity_embeddingA = contextualized_embeddingsA[0][input_idsA[0].index(wordA_vocab_idx)]
                    entity_embeddingB = contextualized_embeddingsB[0][input_idsB[0].index(wordB_vocab_idx)]

                    sim_scores[(word, df.iloc[indexA].sense_def, df.iloc[indexB].sense_def)].append( \
                        cos(torch.FloatTensor(entity_embeddingA), torch.FloatTensor(entity_embeddingB)).item())

print(f'Model: {model_checkpoint} for Dataset: {args.dataset}')

for key, val in sim_scores.items():
    print(key, np.round(torch.mean(torch.Tensor(val)).item(), 2))
