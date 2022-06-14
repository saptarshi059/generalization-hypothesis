from datasets import load_metric
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', type=str)
args = parser.parse_args()

metric = load_metric("squad")

s = pd.read_csv(args.pred_file, header=None)

pred = []
true = []

pred_chars = []
true_chars = []

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

print(metric.compute(predictions=pred, references=true))
print(f'Avg. Gold Len: {np.asarray(true_chars).mean()}')
print(f'Avg. Predicted Len: {np.asarray(pred_chars).mean()}')
