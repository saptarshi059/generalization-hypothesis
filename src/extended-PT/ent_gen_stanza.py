import stanza
from tqdm.auto import tqdm
from datasets import load_dataset
import pickle5 as pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

raw_dataset = args.dataset

if raw_dataset == "Saptarshi7/covid_qa_cleaned_CS":
    nlp = stanza.Pipeline('en', package=None, processors={'ner':['anatem',
                                                            'bc5cdr',
                                                            'bc4chemd',
                                                            'bionlp13cg',
                                                            'jnlpba',
                                                            'linnaeus',
                                                            'ncbi_disease',
                                                            's800',
                                                            'i2b2',
                                                            'radiology'], 'tokenize':'default'})
else:
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')

if args.dataset == 'duorc':
  raw_datasets = load_dataset('duorc', 'SelfRC')
else:
  raw_datasets = load_dataset(args.dataset, use_auth_token=True)

new_ents = []

all_questions = list(set(dataset['train']['question']))

for ques in tqdm(all_questions):
    doc = nlp(ques)
    for ent_dict in doc.entities:
        new_ents.append(ent_dict.text)

with open(f'stanza_ents-from_question-{raw_dataset.replace("/","-")}.pkl', 'wb') as f:
    pickle.dump(new_ents, f)