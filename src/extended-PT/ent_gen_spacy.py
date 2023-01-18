from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import scispacy
import spacy

nlp = spacy.load("en_core_sci_sm")
dataset = load_dataset("Saptarshi7/covid_qa_cleaned_CS", use_auth_token=True)
ents = []

all_questions = dataset['train']['question']

for ques in tqdm(all_questions):
  ents.append([str(x) for x in nlp(ques).ents])

ents_flat = [item for sublist in ents for item in sublist]

with open('spacy_ents-from_question-covidqa.pkl', 'wb') as f:
    pickle.dump(ents_flat, f)