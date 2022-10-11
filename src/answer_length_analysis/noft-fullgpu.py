#!/usr/bin/env python
# coding: utf-8

from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer, logging
from datasets import load_dataset
import csv
import pandas as pd
import torch
import argparse
from tqdm import tqdm

from datetime import datetime
startTime = datetime.now()

logging.set_verbosity(50)

def run_main(full_dataset):
    questions.append(full_dataset['question'])
    gold_answers.append([x['text'][0] for x in full_dataset['answers']])

    if args.dataset in ['duorc']:
        pred_answers.append([x['answer'] for x in nlp(question=full_dataset['question'], context=full_dataset['plot'], handle_impossible_answer=True)])
    elif args.dataset in ['Saptarshi7/techqa-squad-style', 'cuad']:
        pred_answers.append([x['answer'] for x in nlp(question=full_dataset['question'], context=full_dataset['context'], handle_impossible_answer=True)])
    else:
        print('here')
        prediction_dictionaries = nlp(question=full_dataset['question'], context=full_dataset['context'])
        print('predictions done...')
        pred_answers.append([x['answer'] for x in tqdm(prediction_dictionaries)])

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="csarron/roberta-base-squad-v1", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
parser.add_argument('--predictions_file_name', default="roberta-covidqa-preds.csv", type=str)
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
    run_main(raw_datasets['train'])
elif args.dataset in ['squad', 'squad_v2', "Saptarshi7/techqa-squad-style"]:
    run_main(raw_datasets['validation'])
elif args.dataset in ['cuad', 'duorc']:
    run_main(raw_datasets['test'])
    
print('Saving predictions...')
pd.DataFrame(zip(questions, pred_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']).to_pickle(f'{args.predictions_file_name}_predictions.pkl')

print(datetime.now() - startTime)