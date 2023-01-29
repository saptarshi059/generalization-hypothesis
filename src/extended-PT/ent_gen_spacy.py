from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import scispacy
import argparse
import spacy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

spacy.prefer_gpu()
nlp = spacy.load("en_core_sci_sm")
dataset = load_dataset(args.dataset, use_auth_token=True)
ents = []

all_questions = dataset['train']['question']
all_contexts = list(set(dataset['train']['context']))

for ques in tqdm(all_questions):
  ents.append([str(x) for x in nlp(ques).ents])

for ctx in tqdm(all_contexts):
  ents.append([str(x) for x in nlp(ctx).ents])

ents_flat = [item for sublist in ents for item in sublist]

print(f'Total Entities: {len(ents_flat)} | Unique Entities: {len(list(set(ents_flat)))}')

with open('spacy_ents-covidqa.pkl', 'wb') as f:
    pickle.dump(ents_flat, f)