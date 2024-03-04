from datasets import load_dataset
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

if args.dataset == 'duorc':
    raw_datasets = load_dataset('ibm/duorc', 'SelfRC')
else:
    raw_datasets = load_dataset(args.dataset, use_auth_token=True)

for split in tqdm(raw_datasets.keys()):
    questions = []
    contexts = []
    answers = []
    for row in tqdm(raw_datasets[split]):
        questions.append(len(row['question']))
        contexts.append(len(row['context']) if args.dataset != 'duorc' else len(row['plot']))
        if args.dataset != 'duorc':
            if row['answers']['text']:
                answers.extend([len(ans) for ans in row['answers']['text']])
            else:
                answers.append(0)
        else:
            if row['answers']:
                answers.extend([len(ans) for ans in row['answers']])
            else:
                answers.append(0)
    print(
        f'{split} | Number of Records: {raw_datasets[split].num_rows} \n'
        f'| Avg. Question Length: {np.round(np.asarray(questions).mean(), 2)} \n'
        f'| Avg. Context Length: {np.round(np.asarray(contexts).mean(), 2)} \n'
        f'| Avg. Answer Length: {np.round(np.asarray(answers).mean(), 2)}')
