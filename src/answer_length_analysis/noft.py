#!/usr/bin/env python
# coding: utf-8

import argparse
import csv

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer


def run_main():
    questions.append(record['question'])

    try:
        pred_answers.append(nlp(question=record['question'], context=record['context'])['answer'])
    except:
        pred_answers.append("")  # For impossible answers

    try:
        gold_answers.append(record['answers']['text'][0])
    except:
        gold_answers.append("")  # For impossible answers


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/techqa-squad-style", type=str)
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)
raw_datasets = load_dataset(args.dataset, token=True)

gold_answers = []
pred_answers = []
questions = []

if args.dataset == 'Saptarshi7/techqa-squad-style':
    with torch.no_grad():
        for record in tqdm(raw_datasets['validation']):
            run_main()
elif args.dataset in ['cuad', 'ibm/duorc'] :
    with torch.no_grad():
        for record in tqdm(raw_datasets['test']):
            run_main()

print('Saving predictions...')
with open(f'{model_checkpoint.replace("/", "_")}_{args.dataset.replace("/", "_")}_predictions.csv', "w") as f:
    writer = csv.writer(f)
    for row in zip(questions, pred_answers, gold_answers):
        writer.writerow(row)
