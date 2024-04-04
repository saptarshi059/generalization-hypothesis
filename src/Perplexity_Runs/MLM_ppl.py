#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import argparse
import random
import torch
import math


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


def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--corpus_file', default="Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv", type=str)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)

args = parser.parse_args()

g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)

corpus_dataset = load_dataset("csv", data_files=args.corpus_file)
print('Corpus Loaded...')

model_checkpoint = args.model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

tokenized_dataset = corpus_dataset['train'].map(tokenize_function, batched=True, remove_columns=['ent', 'text'])
chunk_size = tokenizer.model_max_length
lm_dataset = tokenized_dataset.map(group_texts, batched=True)
lm_dataset = lm_dataset.remove_columns(["word_ids"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

lm_dataset = lm_dataset.map(insert_random_mask, batched=True, remove_columns=lm_dataset.column_names)
if 'masked_token_type_ids' in lm_dataset.column_names:
    lm_dataset = lm_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
            'masked_token_type_ids': "token_type_ids"
        }
    )
else:
    lm_dataset = lm_dataset.rename_columns(
        {
            "masked_input_ids": "input_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

batch_size = args.batch_size
eval_dataloader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=default_data_collator,
                             worker_init_fn=seed_worker, generator=g)

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

accelerator = Accelerator()
model, optimizer, eval_dataloader = accelerator.prepare(model, optimizer, eval_dataloader)

# Evaluation
model.eval()
losses = []
for step, batch in tqdm(enumerate(eval_dataloader)):
    with torch.no_grad():
        outputs = model(**batch)

    loss = outputs.loss
    losses.append(accelerator.gather(loss.repeat(batch_size)))

losses = torch.cat(losses)
losses = losses[: len(lm_dataset)]
try:
    perplexity = math.exp(torch.mean(losses))
except OverflowError:
    perplexity = float("inf")

print(f'PPL on {args.corpus_file}: {perplexity}')
