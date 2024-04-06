from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from tqdm import tqdm
from datasets import load_dataset
import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default="tiiuae/falcon-7b-instruct", type=str)
    parser.add_argument('--corpus_file',
                        default="../common_terms_freq/Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv", type=str)
    parser.add_argument('--random_state', default=42, type=int)
    args = parser.parse_args()

    # Set random seeds
    g = torch.Generator()
    g.manual_seed(args.random_state)
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    test = load_dataset("csv", data_files=args.corpus_file, split="train")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, device_map='auto', torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    model.eval()
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            set_seed(args.random_state)
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
