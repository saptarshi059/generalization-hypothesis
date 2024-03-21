#!/usr/bin/env python
# coding: utf-8

import argparse
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer, logging, set_seed

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
parser.add_argument('--batch_size', default=40, type=int)
args = parser.parse_args()

g = torch.Generator()
g.manual_seed(42)
torch.manual_seed(42)
random.seed(42)
set_seed(42)

model_checkpoint = args.model_checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)

if args.dataset == 'duorc':
    raw_datasets = load_dataset('duorc', 'SelfRC')
else:
    raw_datasets = load_dataset(args.dataset, use_auth_token=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        question = record['question']
        context = record['context'] if args.dataset != 'duorc' else record['plot']
        answers = record['answers']
        return question, context, answers


# Create custom dataset
dataset = CustomDataset(raw_datasets['test'] if args.dataset in ['ibm/duorc', 'cuad'] else raw_datasets['validation'])
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)

gold_answers = []
pred_answers = []
questions = []

# Run inference in batches
for batch_questions, batch_contexts, batch_gold_answers in tqdm(dataloader):
    pred_batch_answers = nlp(batch_questions, batch_contexts, max_seq_len=384, doc_stride=128,
                             max_answer_length=1000, handle_impossible_answer=True)

    for pred_answer, gold_answer in zip(pred_batch_answers, batch_gold_answers):
        try:
            pred_answers.append(pred_answer['answer'])
        except:
            pred_answers.append("")  # For impossible answers

        try:
            gold_answers.append(gold_answer['answers']['text'] if args.dataset != 'duorc' else gold_answer['answers'])
        except:
            gold_answers.append("")  # For impossible answers

print('Saving predictions...')
pd.DataFrame(zip(questions, pred_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']).to_pickle(
    f'{model_checkpoint.strip("../").replace("/", "_")}_{args.dataset.replace("/", "_")}_predictions.pkl')
