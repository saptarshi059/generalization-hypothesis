#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import QuestionAnsweringPipeline, AutoModelForQuestionAnswering, AutoTokenizer
from datasets import load_dataset
import csv
import torch
import argparse
from tqdm import tqdm

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
raw_datasets = load_dataset(args.dataset, use_auth_token=True)

gold_answers = []
pred_answers = []
questions = []

if args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
    for record in tqdm(raw_datasets['train']):
        run_main()
elif args.dataset == 'squad' or args.dataset == 'squad_v2':
    for record in tqdm(raw_datasets['validation']):
        run_main()
elif args.dataset == 'cuad' or args.dataset == 'duorc':
    for record in tqdm(raw_datasets['test']):
        run_main()

print('Saving predictions...')
with open(f'{model_checkpoint.replace("/","_")}_{args.dataset.replace("/","_")}_predictions.csv', "w") as f:
    writer = csv.writer(f)
    for row in zip(questions, pred_answers, gold_answers):
        writer.writerow(row)