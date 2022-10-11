#!/usr/bin/env python
# coding: utf-8

from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer, logging
from datasets import load_dataset
import csv
import pandas as pd
import torch
import argparse
from tqdm import tqdm
import re

logging.set_verbosity(50)

def run_main():
    questions.append(record['question'])

    try:
        pred_answers.append(nlp(question=record['question'], context=record['context'])['answer'] if args.dataset != 'duorc' else nlp(question=record['question'], context=record['plot'])['answer'])
    except:
        pred_answers.append("") #For impossible answers

    try:
        gold_answers.append(record['answers']['text'] if args.dataset != 'duorc' else record['answers'])
    except:
        gold_answers.append("") #For impossible answers     

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=0)

if args.dataset == 'duorc':
  raw_datasets = load_dataset('duorc', 'SelfRC')
else:
  raw_datasets = load_dataset(args.dataset, use_auth_token=True)

gold_answers = []
pred_answers = []
questions = []

if args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
    for record in tqdm(raw_datasets['train']):
        run_main()
elif args.dataset in ['squad', 'squad_v2', "Saptarshi7/techqa-squad-style"]:
    for record in tqdm(raw_datasets['validation']):
        run_main()
elif args.dataset in ['cuad', 'duorc']:
    for record in tqdm(raw_datasets['test']):
        run_main()

print('Saving predictions...')
if '../' not in model_checkpoint:
    pd.DataFrame(zip(questions, pred_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']).to_pickle(f'{args.model_checkpoint.replace("/","_")}_{args.dataset.replace("/","_")}_predictions.pkl')
else:
    pd.DataFrame(zip(questions, pred_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']).to_pickle(f'{re.search('BERT.*', model_checkpoint).group(0)}_{args.dataset.replace("/","_")}_predictions.pkl')