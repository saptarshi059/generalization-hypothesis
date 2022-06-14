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
  if args.model_checkpoint == 'rc-bidaf':
    pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['best_span_str'])
  else: #For QANet
    try:
      pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['answer']['value'])
    except:
      pred_answers.append("")
  gold_answers.append(record['answers']['text'][0])

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
else:
  for record in tqdm(raw_datasets['validation']):
    run_main()

with open(f'{args.model_checkpoint.replace("/","_")}_{args.dataset.replace("/","_")}_predictions.csv', "w") as f:
    writer = csv.writer(f)
    for row in zip(question, pred_answers, gold_answers):
        writer.writerow(row)