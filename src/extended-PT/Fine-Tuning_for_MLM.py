#!/usr/bin/env python
# coding: utf-8

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler, default_data_collator
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from tqdm import tqdm
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
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
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
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--training_corpus', default="../../../CDQA-v1-whole-entity-approach/data/COVID-QA/wiki_corpus_covidqa_wo_filter.parquet", type=str)
parser.add_argument('--eval_corpus', default="../../../CDQA-v1-whole-entity-approach/src/Utils/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv ", type=str)
parser.add_argument('--trained_model_name', default="distilbert-base-uncased-extended-PT", type=str)
parser.add_argument('--use_new_tokens', default=False, type=str2bool)
parser.add_argument('--random_state', default=42, type=int)
parser.add_argument('--batch_size', default=40, type=int)
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--epochs', default=3, type=int)

args = parser.parse_args()

g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)

model_checkpoint = args.model_checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
chunk_size = tokenizer.model_max_length
batch_size = args.batch_size

#Training Data
train_dataset = load_dataset("parquet", data_files=args.training_corpus)

if ('prompt' in train_dataset.column_names) and ('__index_level_0__' in train_dataset.column_names):
    train_dataset = train_dataset.remove_columns(['prompt', '__index_level_0__'])

if ('entity' in train_dataset.column_names) and ('context' in train_dataset.column_names):
    train_dataset = train_dataset.rename_columns({'entity':'ent', 'context':'text'})

print('Training Corpus Loaded...')

if args.use_new_tokens == True:
    #Adding the new tokens to the vocabulary
    print(f'Original number of tokens: {len(tokenizer)}')
    new_tokens = corpus_dataset['train']['ent']
    tokenizer.add_tokens(new_tokens)
    print(f'New number of tokens: {len(tokenizer)}')

    # The new vector is added at the end of the embedding matrix
    model.resize_token_embeddings(len(tokenizer)) 

train_dataset = train_dataset['train'].map(tokenize_function, batched=True, remove_columns=['ent', 'text'])
train_dataset = train_dataset.map(group_texts, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

train_dataset = train_dataset.remove_columns(["word_ids"])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator, worker_init_fn=seed_worker, generator=g)
print('Training Dataset processed...')

#Eval Data
eval_dataset = load_dataset("csv", data_files=args.eval_corpus)
print('Evaluation Corpus Loaded...')

eval_dataset = eval_dataset['train'].map(tokenize_function, batched=True, remove_columns=['ent', 'text'])
eval_dataset = eval_dataset.map(group_texts, batched=True)
eval_dataset = eval_dataset.remove_columns(["word_ids"])

eval_dataset = eval_dataset.map(insert_random_mask, batched=True, remove_columns=eval_dataset.column_names)
if 'masked_token_type_ids' in eval_dataset.column_names:
    eval_dataset = eval_dataset.rename_columns(
        {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
        'masked_token_type_ids': "token_type_ids"
        }
)
else:
    eval_dataset = eval_dataset.rename_columns(
        {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
        }
)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator, worker_init_fn=seed_worker, generator=g)
print('Evaluation Dataset processed...')

optimizer = AdamW(model.parameters(), lr=args.learning_rate)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

num_train_epochs = args.epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

output_dir = args.trained_model_name

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

print('Training done and model saved')