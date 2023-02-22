from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from collections import Counter
from tqdm.auto import tqdm
import pickle5 as pickle
import scispacy
import argparse
import spacy
import re

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="Saptarshi7/covid_qa_cleaned_CS", type=str)
args = parser.parse_args()

spacy.prefer_gpu(gpu_id=0)
nlp = spacy.load("en_core_sci_scibert")
dataset = load_dataset(args.dataset, use_auth_token=True)
ents = []

all_contexts = list(set(dataset['train']['context']))
all_questions = dataset['train']['question']

ques_ents = []
ctx_ents = []

for ques in tqdm(nlp.pipe(all_questions, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])):
  ques_ents.extend([str(x) for x in nlp(ques).ents])

for ctx in tqdm(nlp.pipe(all_contexts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])):
  ctx_ents.extend([str(x) for x in nlp(ctx).ents])

print(f'Total Entities from Questions: {len(ques_ents)} (Unique: {len(set(ques_ents))})')
print(f'Total Entities from Contexts: {len(ctx_ents)} (Unique: {len(set(ctx_ents))})')
print(f'Total Entities from Questions + Contexts: {len(ques_ents + ctx_ents)} (Unique: {len(set(ques_ents + ctx_ents))})')

total_ents = list(set(list(set(ctx_ents)) + list(set(ques_ents))))
total_ents_filtered = []

for ent in total_ents:
  if (ent.isnumeric() == False) \
  and (len(ent) > 5) \
  and not (re.search((r'https*|doi:|\[\d+\]|[\u0080-\uFFFF]|et al.|[Aa]uthor|www|\n|\*|\||@|;|&|!'), ent)) \
  and not (re.search(r'Fig[ures]*', ent, re.IGNORECASE)) \
  and (ent.count('(') == ent.count(')')) \
  and (ent.count('"') % 2 == 0) \
  and (ent.count('\'') % 2 == 0) \
  and (ent.count('[') == ent.count(']')):
    total_ents_filtered.append(ent)

vocab = {v: k for k, v in dict(enumerate(total_ents_filtered)).items()}

print(f'Size of filtered & unweighted vocab: {len(vocab)}')

print('Weighting each entity by their IDF score and choosing top 25000 entities...')
vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words='english', lowercase=False)
X = vectorizer.fit_transform(all_contexts + all_questions)

idf = vectorizer.idf_
top_N_ents = []
for k, _ in {k: v for k, v in sorted(dict(zip(vectorizer.get_feature_names_out(), idf)).items(), key=lambda item: item[1], reverse=True)}.items():
  top_N_ents.append(k)
  if len(top_N_ents) == 25000:
    break

print('Saving entity file...')

with open('spacy_ents-covidqa.pkl', 'wb') as f:
    pickle.dump(top_N_ents, f)