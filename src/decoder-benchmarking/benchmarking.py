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
        for row in tqdm(ds):
            context = row['context'] if args.dataset != 'ibm/duorc' else row['plot']
            context_chunks = tokenizer(context, add_special_tokens=False, truncation=True, max_length=400,
                                       stride=200, return_overflowing_tokens=True)
            true_spans = row['answers']['text'] if args.dataset != 'ibm/duorc' else row['answers']
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


class TechQA(Dataset):
    def __init__(self, ds, prompt):
        self.samples = []
        for row in tqdm(ds):
            context = row['context']
            context_chunks = tokenizer(context, add_special_tokens=False, truncation=True, max_length=1500,
                                       stride=200, return_overflowing_tokens=True)
            true_spans = row['answers']['text']
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











class NoChunkDataset(Dataset):
    def __init__(self, ds, prompt):
        self.samples = []
        for row in tqdm(ds):
            self.samples.append(prompt.format(context=row['context'], question=row['question'])
                                if args.dataset == 'squad' else
                                prompt.format(context=row['plot'], question=row['question']))

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'ibm/duorc':
        dataset = load_dataset('ibm/duorc', 'SelfRC')
    else:
        dataset = load_dataset(args.dataset, token=True, trust_remote_code=True)

    # Changing dataset split names for consistency
    if args.dataset in ['squad', 'Saptarshi7/techqa-squad-style']:
        dataset['test'] = dataset.pop('validation')
    elif args.dataset in ['cuad', 'ibm/duorc']:
        pass  # Since they already contain a test split
    elif args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
        dataset['test'] = dataset.pop('train')

    # Keeping only answerable questions for TechQA & DuoRC
    if args.dataset == 'Saptarshi7/techqa-squad-style':
        dataset['test'] = dataset['test'].filter(lambda x: x['answers']['text'] != [])
    elif args.dataset == 'ibm/duorc':
        dataset['test'] = dataset['test'].filter(lambda x: x['no_answer'] is False)

    if args.dataset in ['squad', 'ibm/duorc']:
        formatted_dataset = NoChunkDataset(dataset['test'], args.prompt)
    elif args.dataset == 'Saptarshi7/techqa-squad-style':
        formatted_dataset = TechQA(dataset['test'], args.prompt)
    dataloader = DataLoader(formatted_dataset, batch_size=args.batch_size, shuffle=False)

    c = 0
    import re
    for expanded_prompt, true_answers in zip(iter(formatted_dataset), dataset['test']['answers']):
        for ans in true_answers['text'] if args.dataset != 'ibm/duorc' else true_answers:
            if not re.search(fr'{re.escape(ans)}', expanded_prompt, re.IGNORECASE):
                print(expanded_prompt, ans)
                print('>>>>>>>>>>>>>>>>>>>>>>')
                c += 1
                break
    print(f'No. of context chunks NOT containing the respective answer span: {c}')
    if c != 0:
        exit('Exited program because of inconsistent number of samples...')
    else:
        exit('Dataset can be processed correctly by this model...')

    gold_answers = []
    for el in dataset['test']['answers']:
        gold_answers.append(el['text'])

    # Loading models start here
    if checkpoint == 'tiiuae/falcon-7b-instruct':
        generator = pipeline("text-generation", model=checkpoint, tokenizer=tokenizer, torch_dtype=torch.bfloat16,
                             trust_remote_code=True, device_map="auto", pad_token_id=tokenizer.eos_token_id)
        print('The Falcon has landed... ;)')
    else:
        generator = pipeline('text-generation', model=checkpoint, tokenizer=tokenizer, device_map='auto')
        print(f'Model: {checkpoint} loaded...')

    print('Generating Predictions...')
    predictions = []
    # Using Flash Attention...
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        for batch in tqdm(dataloader):
            generations = generator(batch, max_new_tokens=50)
            predictions.extend([x[0]['generated_text'].split('Answer: ')[1].strip() for x in generations])

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
