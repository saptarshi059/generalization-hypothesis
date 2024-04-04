import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import random

def tokenize_and_process_chunks(tokenizer, model, chunk_texts, device):
    """Tokenize and process chunks of texts."""
    tokenized_texts = tokenizer(chunk_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = tokenized_texts.input_ids.to(device)
    labels = input_ids.clone()
    labels[:, :-1] = -100  # Set unwanted tokens to -100 in-place

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        neg_log_likelihood = outputs.loss
    return neg_log_likelihood

def main(args):
    # Set random seeds
    torch.manual_seed(args.random_state)
    random.seed(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint).to(device)

    # Load dataset
    corpus_dataset = load_dataset("csv", data_files=args.corpus_file)
    texts = "\n\n".join(corpus_dataset['train']["text"])

    # Tokenize input in chunks
    chunk_size = 10000
    chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]

    # Process chunks in batches
    batch_size = 8
    nlls = []
    for i in tqdm(range(0, len(chunks), batch_size)):
        chunk_batch = chunks[i:i+batch_size]
        neg_log_likelihood = tokenize_and_process_chunks(tokenizer, model, chunk_batch, device)
        nlls.append(neg_log_likelihood)

    # Compute PPL
    avg_nll = torch.stack(nlls).mean()
    ppl = torch.exp(avg_nll)
    print(f'PPL of {args.model_checkpoint} on {args.corpus_file}: {ppl}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', default="tiiuae/falcon-7b-instruct", type=str)
    parser.add_argument('--corpus_file', default="Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv", type=str)
    parser.add_argument('--random_state', default=42, type=int)
    args = parser.parse_args()

    main(args)
