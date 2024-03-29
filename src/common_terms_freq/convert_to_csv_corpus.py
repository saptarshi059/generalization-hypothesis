from datasets import load_dataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

print('Reading dataset...')
if args.dataset == 'duorc':
  dataset = load_dataset('duorc', 'SelfRC')
  total = list(set(dataset['train']['plot'])) + list(set(dataset['train']['question']))
else:
  dataset = load_dataset(args.dataset, token=True)
  total = list(set(dataset['train']['context'])) + list(set(dataset['train']['question']))

pd.DataFrame(zip(list(range(len(total))), total), columns=['ent', 'text']).to_csv(f"{args.dataset.replace('/','-')}_for_PPL_eval.csv", index=False)
print('Saving dataset...')
