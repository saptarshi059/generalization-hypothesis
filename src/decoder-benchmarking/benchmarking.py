import argparse

import pandas as pd
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


class QADataset(Dataset):
    def __init__(self, ds, prompt):
        self.samples = []
        for row in tqdm(ds['test']):
            context = row['context'] if ds != 'ibm/duorc' else row['plot']
            context_chunks = tokenizer(context, add_special_tokens=False, truncation=True, max_length=400,
                                       stride=200, return_overflowing_tokens=True)
            true_spans = row['answers']['text'] if ds != 'ibm/duorc' else row['answers']
            question = row['question']

            flag = 0
            for chunk in context_chunks['input_ids']:
                decoded_chunk = tokenizer.decode(chunk, clean_up_tokenization_spaces=False)
                for ans in true_spans:
                    if ans in decoded_chunk:
                        flag = 1
                    else:
                        flag = 0
                        break
                if flag == 1:
                    self.samples.append(prompt.format(context=decoded_chunk, question=question))
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Saptarshi7/covid_qa_cleaned_CS')
    parser.add_argument('--model_checkpoint', default='medalpaca/medalpaca-7b')
    parser.add_argument('--prompt', default='Context: {context}\n\nQuestion: {question}\n\nAnswer: ')
    parser.add_argument('--batch_size', default=40)
    args = parser.parse_args()

    checkpoint = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if args.dataset == 'ibm/duorc':
        dataset = load_dataset('ibm/duorc', 'SelfRC')
    else:
        dataset = load_dataset(args.dataset, token=True)

    # Changing dataset split names for consistency
    if args.dataset in ['squad', 'Saptarshi7/techqa-squad-style']:
        dataset['test'] = dataset.pop('validation')
    elif args.dataset in ['cuad', 'ibm/duorc']:
        pass  # Since they already contain a test split
    elif args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
        dataset['test'] = dataset.pop('train')

    formatted_dataset = QADataset(dataset, args.prompt)
    dataloader = DataLoader(formatted_dataset, batch_size=args.batch_size, shuffle=False)

    c = 0
    for i, j in zip(iter(formatted_dataset), dataset['train']['answers']):
        if j['text'][0] not in i:
            c += 1
    print(f'No. of context chunks NOT containing the respective answer span: {c}')
    if c != 0:
        exit('Exited program because of inconsistent number of samples...')

    gold_answers = []
    for el in dataset['train']['answers']:
        gold_answers.append(el['text'])

    # Loading models start here
    if checkpoint == 'tiiuae/falcon-7b-instruct':
        generator = pipeline("text-generation", model=checkpoint, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
                             trust_remote_code=True, device_map="auto", pad_token_id=tokenizer.eos_token_id)
        print('The Falcon has landed... ;)')
    else:
        generator = pipeline('text-generation', model=checkpoint, tokenizer=tokenizer, device_map='auto')

    predictions = []
    for batch in tqdm(dataloader):
        generations = generator(batch, max_new_tokens=50)
        predictions.extend([x[0]['generated_text'].split(args.answer_prompt)[1].strip() for x in generations])

    print('Computing Scores...')
    metric = load_metric('squad')

    formatted_predictions = []
    formatted_gold = []

    for ID, (pred, gold) in enumerate(zip(predictions, gold_answers)):
        formatted_predictions.append({"id": str(ID), "prediction_text": pred})
        formatted_gold.append({"id": str(ID), "answers": {'answer_start': [1 for i in range(len(gold))], 'text': gold}})

    metrics = metric.compute(predictions=formatted_predictions, references=formatted_gold)
    print(metrics)

    print('Saving predictions...')
    pd.DataFrame(zip(predictions, gold_answers),
                 columns=['predictions', 'reference']).to_pickle(f'{checkpoint.replace("/", "_")}_preds.pkl')

    count = 0
    for (pred, ctx) in zip(predictions, dataset['train']['context']):
        if pred in ctx:
            count += 1
    print(f'No. of predictions ACTUALLY (exactly) IN the entire context: {count}')