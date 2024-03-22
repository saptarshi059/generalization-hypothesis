# -*- coding: utf-8 -*-

import argparse
import collections
import random

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, logging, default_data_collator, \
    set_seed


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


def prepare_validation_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    if args.dataset != 'ibm/duorc':
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    else:
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["plot" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"] if args.dataset != 'ibm/duorc' else example["plot"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()
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
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            if impossible_questions:
                predicted_answers.append({"id": example_id,
                                          "prediction_text":
                                              best_answer["text"] if args.dataset != 'ibm/duorc' else best_answer,
                                          "no_answer_probability": 0.0})
            else:
                predicted_answers.append({"id": example_id,
                                          "prediction_text":
                                              best_answer["text"] if args.dataset != 'ibm/duorc' else best_answer})
        else:
            if impossible_questions:
                predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability": 1.0})
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


parser = argparse.ArgumentParser()

parser.add_argument('--impossible_questions', default=True, type=str2bool)
parser.add_argument('--dataset', default='Saptarshi7/techqa-squad-style', type=str)
parser.add_argument('--model_checkpoint', default="distilbert-base-uncased", type=str)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_length', default=384, type=int)
parser.add_argument('--stride', default=128, type=int)
parser.add_argument('--n_best', default=20, type=int)
parser.add_argument('--max_answer_length', default=30, type=int)
parser.add_argument('--random_state', default=42, type=int)

args = parser.parse_args()

logging.set_verbosity(50)

# Initializing all random number methods.
g = torch.Generator()
g.manual_seed(args.random_state)
torch.manual_seed(args.random_state)
random.seed(args.random_state)
set_seed(args.random_state)

impossible_questions = args.impossible_questions
model_checkpoint = args.model_checkpoint
batch_size = args.batch_size
accelerator = Accelerator()
device = accelerator.device
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = default_data_collator
max_length = args.max_length  # The maximum length of a feature (question and context)
doc_stride = args.stride  # The authorized overlap between two part of the context when splitting it is needed.
max_answer_length = args.max_answer_length
n_best = args.n_best
pad_on_right = tokenizer.padding_side == "right"

if args.dataset == 'ibm/duorc':
    raw_datasets = load_dataset('ibm/duorc', 'SelfRC')
else:
    raw_datasets = load_dataset(args.dataset, token=True, trust_remote_code=True)

if args.dataset == 'Saptarshi7/techqa-squad-style':  # Only has validation split.
    validation_dataset = raw_datasets['validation'].map(prepare_validation_features, batched=True,
                                                        remove_columns=raw_datasets['validation'].column_names)
# CUAD/DuoRC - both have test splits.
else:
    validation_dataset = raw_datasets['validation'].map(prepare_validation_features, batched=True,
                                                  remove_columns=raw_datasets['validation'].column_names)

if impossible_questions:
    metric = load("squad_v2")
else:
    metric = load("squad")

validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")
eval_dataloader = DataLoader(validation_set, collate_fn=data_collator, batch_size=batch_size,
                             worker_init_fn=seed_worker, generator=g)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

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

metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets['validation'])
