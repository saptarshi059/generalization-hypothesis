# -*- coding: utf-8 -*-

#!pip install allennlp-models
#!pip install --upgrade google-cloud-storage

from allennlp_models import pretrained
from datasets import load_dataset
from tqdm import tqdm
import argparse
import csv

def run_main():
  question.append(record['question'])
  
  try:
    gold_answers.append(record['answers']['text'][0])
  except:
    gold_answers.append("") #For impossible questions.

  if args.model_checkpoint == 'rc-bidaf':
    
    try:
      pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['best_span_str'])
    except:
      pred_answers.append("")

  else: #For QANet
    
    try:
      pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['answer']['value'])
    except:
      pred_answers.append("")
  
parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="rc-bidaf", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

model = pretrained.load_predictor(args.model_checkpoint, cuda_device=0)

#!huggingface-cli login

raw_datasets = load_dataset(args.dataset, use_auth_token=True)

gold_answers = []
pred_answers = []
question = []

if args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
  for record in tqdm(raw_datasets['train']):
    run_main()
elif args.dataset == 'squad' or args.dataset == 'squad_v2':
    for record in tqdm(raw_datasets['validation']):
        run_main()
elif args.dataset == 'cuad':
    for record in tqdm(raw_datasets['test']):
        run_main()

with open(f'{args.model_checkpoint.replace("/","_")}_{args.dataset.replace("/","_")}_predictions.csv', "w") as f:
    writer = csv.writer(f)
    for row in zip(question, pred_answers, gold_answers):
        writer.writerow(row)