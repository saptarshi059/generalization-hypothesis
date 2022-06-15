from datasets import load_metric
import pandas as pd
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
parser.add_argument('--metric', default='squad' ,type=str)
args = parser.parse_args()

metric = load_metric(args.metric)

s = pd.read_csv(args.pred_file, header=None)

pred = []
true = []

pred_chars = []
true_chars = []

if args.metric == 'squad':
  for row in s.itertuples():
    try:
      pred.append({"id": str(row.Index), "prediction_text": row._2})
    except:
      pred.append("")

    try:
      pred_chars.append(len(row._2))
    except:
      pred_chars.append(0)

    true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [row._3]}})  
    true_chars.append(len(row._3))

elif args.metric == 'squad_v2':
  for row in s.itertuples():
    if type(row._2) == float and math.isnan(row._2):
      pred.append({"id": str(row.Index), "prediction_text": "", 'no_answer_probability': 1.})
      pred_chars.append(0)
    else:
      pred.append({"id": str(row.Index), "prediction_text": row._2, 'no_answer_probability': 0.})
      pred_chars.append(len(row._2))
      
    if type(row._3) == float and math.isnan(row._3):
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [""]}})
      true_chars.append(0)
    else:
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [row._3]}})
      true_chars.append(len(row._3))

elif args.metric == 'cuad':
  for row in s.itertuples():
    if type(row._2) == float and math.isnan(row._2):
      pred.append({"id": str(row.Index), "prediction_text": [" "]})
      pred_chars.append(0)
    else:
      pred.append({"id": str(row.Index), "prediction_text": [row._2]})
      pred_chars.append(len(row._2))
      
    if type(row._3) == float and math.isnan(row._3):
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [" "]}})
      true_chars.append(0)
    else:
      true.append({"id": str(row.Index), "answers": {'answer_start': [1], 'text': [row._3]}})
      true_chars.append(len(row._3))
    
print(metric.compute(predictions=pred, references=true))
print(f'Avg. Gold Len: {np.asarray(true_chars).mean()}')
print(f'Avg. Predicted Len: {np.asarray(pred_chars).mean()}')
