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


class CLMDataset(Dataset):
    def __init__(self, text_chunks):
        self.chunks = []
        for chunk in text_chunks:
            tokenized_texts = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = tokenized_texts['input_ids'].to(device)
            labels = input_ids.clone()
            labels[:, :-1] = -100  # Set unwanted tokens to -100 in-place
            self.chunks.append((input_ids, labels))

    def __getitem__(self, idx):
        return self.chunks[idx]

    def __len__(self):
        return len(self.chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default="tiiuae/falcon-7b-instruct", type=str)
    parser.add_argument('--corpus_file', default="Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv",
                        type=str)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
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
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint).to(device)
    model.eval()

    # Load dataset
    corpus_dataset = load_dataset("csv", data_files=args.corpus_file)
    texts = "\n\n".join(corpus_dataset['train']["text"])
    chunk_size = 10000
    chunk_dataset = CLMDataset([texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)])
    chunk_dataset_dataloader = DataLoader(chunk_dataset, shuffle=False, batch_size=batch_size,
                                          worker_init_fn=seed_worker, generator=g)

    nlls = []
    for input_ids, labels in tqdm(chunk_dataset_dataloader):
        with torch.no_grad():
            set_seed(args.random_state)
            outputs = model(input_ids, labels=labels)
            nlls.append(outputs.loss)

    # Compute PPL
    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll)
    print(f'PPL of {args.model_checkpoint} on {args.corpus_file}: {ppl}')
