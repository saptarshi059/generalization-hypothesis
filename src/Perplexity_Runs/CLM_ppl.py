import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TextDataset(Dataset):
    def __init__(self, texts):
        self.input_ids = []
        self.attention_mask = []
        for sample in tqdm(texts):
            tokenized_chunks = tokenizer(sample, return_tensors="pt", padding='max_length', truncation=True,
                                         max_length=max_length, stride=64, return_overflowing_tokens=True)
            self.input_ids.extend(tokenized_chunks['input_ids'])
            self.attention_mask.extend(tokenized_chunks['attention_mask'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default="tiiuae/falcon-7b-instruct", type=str)
    parser.add_argument('--corpus_file', default="Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv",
                        type=str)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    # Set random seeds
    g = torch.Generator()
    g.manual_seed(args.random_state)
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, torch_dtype=torch.float16,
                                                 device_map="auto")
    model.eval()

    test_dataset = load_dataset("csv", data_files=args.corpus_file, split='train')
    texts = test_dataset["text"]

    max_length = 1024  # Otherwise it's freaking out on other models. - Falcon fine, others not.

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, worker_init_fn=seed_worker, generator=g, shuffle=False)

    nlls = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = input_ids.clone()
            labels[:, :-1] = -100
            set_seed(args.random_state)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).mean())
    print(f'Perplexity of model {args.model_checkpoint} on dataset {args.corpus_file}: {ppl.item()}')
