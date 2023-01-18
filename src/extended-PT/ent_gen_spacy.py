from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import scispacy
import argparse
import spacy

parser = argparse.ArgumentParser()
parser.add_argument('--top_N_ents', default=500, type=int)
args = parser.parse_args()

raw_datasets = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)
ents = []

all_questions = dataset['train']['question']

for row in tqdm(all_questions):
  ents.append([str(x) for x in nlp(row['question']).ents])

ents_flat = [item for sublist in ents for item in sublist]

selected_ents = [x[0] for x in Counter(ents_flat).most_common()[:args.top_N_ents]]

with open('spacy_ents-from_question-covidqa.pkl', 'wb') as f:
    pickle.dump(selected_ents, f)