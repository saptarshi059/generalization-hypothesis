#!/usr/bin/env python
# coding: utf-8

import argparse
import csv

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        question = record['question']
        context = record['context'] if 'context' in record else record['plot']
        return question, context


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/techqa-squad-style", type=str)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0, torch_dtype=torch.float16)
raw_datasets = load_dataset(args.dataset, token=True)

gold_answers = []
pred_answers = []
questions = []

if args.dataset == 'Saptarshi7/techqa-squad-style':
    dataset = QADataset(raw_datasets['validation'].filter(lambda x: x['question'] != ''), tokenizer)
elif args.dataset in ['cuad', 'ibm/duorc']:
    dataset = QADataset(raw_datasets['test'], tokenizer)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

gold_answers.extend([x for x in dataset['answers']['text'][0]])

with torch.no_grad():
    for questions, contexts in tqdm(dataloader):
        predictions = nlp(question=list(questions), context=list(contexts), handle_impossible_answer=True)
        try:
            pred_answers.append(predictions['answer'])
        except:
            pred_answers.append("")  # For impossible answers

print('Saving predictions...')
with open(f'{model_checkpoint.replace("/", "_")}_{args.dataset.replace("/", "_")}_predictions.csv', "w") as f:
    writer = csv.writer(f)
    for row in zip(questions, pred_answers, gold_answers):
        writer.writerow(row)
