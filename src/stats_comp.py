from datasets import load_dataset
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

if args.dataset == 'duorc':
  raw_datasets = load_dataset('duorc', 'SelfRC')
else:
  raw_datasets = load_dataset(args.dataset, use_auth_token=True)

for split in raw_datasets.keys():
	q = []
	c = []
	for row in raw_datasets[split]:
		q.append(len(row['question']))
		c.append(len(row['context']))
	print(f'{split} | Avg. question Length: {np.asarray(q).mean()} | Avg. Context Length: {np.asarray(c).mean()}')