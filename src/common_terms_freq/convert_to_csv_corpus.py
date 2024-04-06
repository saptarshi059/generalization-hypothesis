from datasets import load_dataset
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
    parser.add_argument('--split', default='train', type=str)
    args = parser.parse_args()

    print('Reading dataset...')
    if args.dataset == 'duorc':
        dataset = load_dataset('duorc', 'SelfRC')
        total = list(set(dataset[args.split]['plot'])) + list(set(dataset[args.split]['question']))
    else:
        dataset = load_dataset(args.dataset, token=True)
        total = list(set(dataset[args.split]['context'])) + list(set(dataset[args.split]['question']))

    pd.DataFrame(zip(list(range(len(total))), total), columns=['ent', 'text']).to_csv(
        f"{args.dataset.replace('/', '-')}_{args.split}_for_PPL_eval.csv", index=False)
    print('Saving dataset...')
