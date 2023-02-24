# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoConfig, logging, default_data_collator, get_scheduler
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from timm.optim import Lamb
from tqdm.auto import tqdm
from evaluate import load
import transformers
import numpy as np
import collections
import argparse
import random
import torch

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

def encode(example, encoder_max_len=250, decoder_max_len=54):
    context = example['context']
    question = example['question']
    answer = example['answers']['text']
  
    question_plus = f"answer_me: {str(question)}"
    question_plus += f" context: {str(context)} </s>"
    
    answer_plus = ', '.join([i for i in list(answer)])
    answer_plus = f"{answer_plus} </s>"
    
    encoder_inputs = tokenizer(question_plus, truncation=True, 
                               return_tensors='pt', max_length=encoder_max_len,
                              pad_to_max_length=True)
    
    decoder_inputs = tokenizer(answer_plus, truncation=True, 
                               return_tensors='pt', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

parser = argparse.ArgumentParser()

parser.add_argument('--squad_version2', default=False, type=str2bool)
parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--trained_model_name', default="distilbert-base-uncased-squad", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_length', default=384, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--learning_rate', default=2e-5, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--epochs', default=3, type=int)
parser.add_argument('--n_best', default=20, type=int)
parser.add_argument('--max_answer_length', default=30, type=int)
parser.add_argument('--trial_mode', default=False, type=str2bool)
parser.add_argument('--random_state', default=42, type=int)

args = parser.parse_args()

logging.set_verbosity(50)

# Initializing all random number methods.
g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)

# This flag is the difference between SQUAD v1 or 2 (if you're using another dataset, it indicates if impossible
# answers are allowed or not).
squad_v2 = args.squad_version2
model_checkpoint = args.model_checkpoint
batch_size = args.batch_size

accelerator = Accelerator()
device = accelerator.device

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

data_collator = default_data_collator

max_length = args.max_length # The maximum length of a feature (question and context)
doc_stride = args.stride # The authorized overlap between two part of the context when splitting it is needed.
max_answer_length = args.max_answer_length
n_best = args.n_best

if args.trial_mode == True:
    print('Running Code in Trial Mode to see if everything works properly...')
    raw_datasets = load_dataset("squad_v2" if squad_v2 else "squad", split=['train[:160]','validation[:10]']) #Testing purposes
    train_dataset = raw_datasets[0].map(encode
    validation_dataset = raw_datasets[1].map(encode)
else:
    raw_datasets = load_dataset("squad_v2" if squad_v2 else "squad")
    train_dataset = raw_datasets['train'].map(encode, batched=True)
    validation_dataset = raw_datasets['validation'].map(encode, batched=True)

metric = load("squad")

train_dataset.set_format("torch")
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)

validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
eval_dataloader = DataLoader(validation_set, collate_fn=data_collator, batch_size=batch_size, worker_init_fn=seed_worker, generator=g)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
output_dir = args.trained_model_name

optimizer = Lamb(model.parameters(), lr=args.learning_rate)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)
model.to(device)

num_train_epochs = args.epochs
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("Evaluation!")
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    if args.trial_mode == True:
        metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets[1])
    else:
        metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets['validation'])
    
    print(f"epoch {epoch}:", metrics)

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        #repo.push_to_hub(commit_message=f"Training in progress epoch {epoch}", blocking=False)