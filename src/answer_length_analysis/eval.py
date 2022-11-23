from datasets import load_metric
import pickle5 as pickle
import pandas as pd
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
parser.add_argument('--metric', default='squad' ,type=str)
args = parser.parse_args()

metric = load_metric(args.metric)

s = pickle.load(open(args.pred_file, 'rb'))

pred = []
true = []

pred_chars = []
true_chars = []

if args.metric == 'squad': # Use squad for covid-qa as well
  for row in s.itertuples():
    try:
      pred.append({"id": str(row.Index), "prediction_text": row.predictions})
    except:
      pred.append("")

    try:
      pred_chars.append(len(row.predictions))
    except:
      pred_chars.append(0)

    true.append({"id": str(row.Index), "answers": {'answer_start': [1 for i in range(len(row.gold_answers))], 'text': row.gold_answers}})  
    true_chars.append([len(x) for x in row.gold_answers])

elif args.metric == 'squad_v2': # Use SQuADv2 for DuoRC & techqa as well
  for row in s.itertuples():
    if len(str(row.predictions)) == 0:
      pred.append({"id": str(row.Index), "prediction_text": "", 'no_answer_probability': 1.})
      pred_chars.append(0)
    else:
      pred.append({"id": str(row.Index), "prediction_text": row.predictions, 'no_answer_probability': 0.})
      pred_chars.append(len(str(row.predictions)))
      
    if len(row.gold_answers) == 0:
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': []}})
      true_chars.append([0])
    else:
      true.append({"id": str(row.Index), "answers": {'answer_start': [1 for i in range(len(row.gold_answers))], 'text': row.gold_answers}})  
      true_chars.append([len(x) for x in row.gold_answers])

elif args.metric == 'cuad':
  for row in s.itertuples():
    if type(row.predictions) == float and math.isnan(row.predictions):
      pred.append({"id": str(row.Index), "prediction_text": [" "]})
      pred_chars.append(0)
    else:
      pred.append({"id": str(row.Index), "prediction_text": [row.predictions]})
      pred_chars.append(len(row.predictions))
      
    if type(row.gold_answers) == float and math.isnan(row.gold_answers):
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [" "]}})
      true_chars.append([0])
    else:
      true.append({"id": str(row.Index), "answers": {'answer_start': [1 for i in range(len(row.gold_answers))], 'text': row.gold_answers}})
      true_chars.append([len(x) for x in row.gold_answers])

true_chars = [item for sublist in true_chars for item in sublist]
    
print(metric.compute(predictions=pred, references=true))
print(f'Avg. Gold Len: {np.round(np.asarray(true_chars).mean(),2)}')
print(f'Avg. Predicted Len: {np.round(np.asarray(pred_chars).mean(),2)}')
