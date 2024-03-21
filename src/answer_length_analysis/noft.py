#!/usr/bin/env python
# coding: utf-8

import argparse

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer, logging

logging.set_verbosity(50)

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
parser.add_argument('--batch_size', default=40, type=int)
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)

if args.dataset == 'duorc':
    raw_datasets = load_dataset('duorc', 'SelfRC')
else:
    raw_datasets = load_dataset(args.dataset, use_auth_token=True)


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
dataloader = DataLoader(dataset, batch_size=args.batch_size)

gold_answers = []
pred_answers = []
questions = []


# Run inference in batches
for batch_questions, batch_contexts, batch_gold_answers in tqdm(dataloader):
    pred_batch_answers = nlp(batch_questions, batch_contexts)  # Perform inference on batch

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
