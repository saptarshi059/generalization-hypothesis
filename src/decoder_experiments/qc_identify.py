import argparse

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, set_seed


class ChunkDataset(Dataset):
    def __init__(self, ds):
        self.samples = []
        for row in tqdm(ds):
            context = row['context']
            context_chunks = tokenizer(context, add_special_tokens=False, truncation=True, max_length=400, stride=200,
                                       return_overflowing_tokens=True)

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
                    chat = [{"role": "user",
                             "content": f"Write the context and question exactly.\nContext: {decoded_chunk}"
                                        f"\nQuestion: {question}"}]
                    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    final_tuple = (question, decoded_chunk, prompt)
                    self.samples.append(final_tuple)
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NoChunkDataset(Dataset):
    def __init__(self, ds):
        self.samples = []
        for row in tqdm(ds):
            question = row['question']
            context = row['context'] if args.dataset == 'squad' else row['plot']
            chat = [{"role": "user",
                     "content": f"Write the context and question exactly.\nContext: {context}"
                                f"\nQuestion: {question}"}]
            prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            final_tuple = (question, context, prompt)
            self.samples.append(final_tuple)

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
    parser.add_argument('--batch_size', type=int, default=40)
    args = parser.parse_args()

    set_seed(43)

    checkpoint = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'ibm/duorc':
        dataset = load_dataset('ibm/duorc', 'SelfRC')
    else:
        dataset = load_dataset(args.dataset, token=True, trust_remote_code=True)

    # Changing dataset split names for consistency
    if args.dataset == 'squad':
        dataset['test'] = dataset.pop('validation')
    elif args.dataset == 'Saptarshi7/covid_qa_cleaned_CS':
        dataset['test'] = dataset.pop('train')

    # Keeping only answerable questions for TechQA/DuoRC/CUAD
    if args.dataset == 'ibm/duorc':
        dataset['test'] = dataset['test'].filter(lambda x: x['no_answer'] is False)

    if args.dataset in ['squad', 'ibm/duorc']:
        formatted_dataset = NoChunkDataset(dataset['test'])
    else:
        formatted_dataset = ChunkDataset(dataset['test'])
    dataloader = DataLoader(formatted_dataset, batch_size=args.batch_size, shuffle=False)

    for batch_question, batch_ctx, batch_prompts in tqdm(dataloader):
        print(batch_question)
        print('............................................')
        print(batch_ctx)
        print('............................................')
        print(batch_prompts)
        print('............................................')
        break

    '''
    generator = pipeline('text-generation', model=checkpoint, tokenizer=tokenizer, device='cuda:0',
                         pad_token_id=tokenizer.eos_token_id, torch_dtype=torch.bfloat16)
    print(f'Model: {checkpoint} loaded...')

    gold_answers = []
    for el in dataset['test']['answers']:
        gold_answers.append(el['text'] if args.dataset != 'ibm/duorc' else el)

    gold_answers = [x['text'] for x in dataset['test']['answers'][:2]]

    print('Generating Predictions...')
    predictions = []
    for batch in tqdm(dataloader):
        generations = generator(batch, max_new_tokens=1500, renormalize_logits=True)
        predictions.extend([x[0]['generated_text'] for x in generations])
        break

    print('Saving predictions...')
    pd.DataFrame(zip(predictions, gold_answers),
                 columns=['predictions', 'reference']).to_pickle(f'{checkpoint.replace("/", "_")}'
                                                                 f'_{args.dataset}_preds.pkl')
    '''