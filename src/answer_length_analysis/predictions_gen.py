# -*- coding: utf-8 -*-

#!pip install allennlp-models
#!pip install --upgrade google-cloud-storage

from allennlp_models import pretrained
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import argparse
import csv

def run_main():
  question.append(record['question'])
  
  try:
    gold_answers.append(record['answers']['text'] if args.dataset != 'duorc' else record['answers'])
  except:
    gold_answers.append("") #For impossible answers 

  if args.model_checkpoint == 'rc-bidaf':
    
    try:
      pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['best_span_str'] if args.dataset != 'duorc' else model.predict_json({"passage": record['plot'], "question": record['question']})['best_span_str'])
    except:
      pred_answers.append("")

  else: #For QANet
    
    try:
      pred_answers.append(model.predict_json({"passage": record['context'], "question": record['question']})['answer']['value'] if args.dataset != 'duorc' else model.predict_json({"passage": record['plot'], "question": record['question']})['answer']['value'])
    except:
      pred_answers.append("")
  
parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="rc-bidaf", type=str)
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

model = pretrained.load_predictor(args.model_checkpoint, cuda_device=0)

#!huggingface-cli login

if args.dataset == 'duorc':
  raw_datasets = load_dataset('duorc', 'SelfRC')
else:
  raw_datasets = load_dataset(args.dataset, use_auth_token=True)

gold_answers = []
pred_answers = []
question = []

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
pd.DataFrame(zip(question, pred_answers, gold_answers), columns=['question', 'predictions', 'gold_answers']).to_pickle(f'{args.model_checkpoint.replace("/","_")}_{args.dataset.replace("/","_")}_predictions.pkl')