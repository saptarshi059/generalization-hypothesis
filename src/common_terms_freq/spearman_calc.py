from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from tqdm.auto import tqdm
from scipy import stats
import pandas as pd
import argparse
import spacy


def find_unigrams(texts):
    all_unigrams = []

    nlp = spacy.load("en_core_web_sm")
    for doc in tqdm(nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", 'senter', "attribute_ruler",
                                             "lemmatizer", 'ner'], batch_size=10_000), total=len(texts)):
        for token in doc:
            if (not token.is_stop) and (not token.is_punct) and (not token.is_bracket) and \
                    (not token.is_quote) and \
                    (not token.is_currency) and \
                    (not token.like_url) and \
                    (not token.like_num) and \
                    (not token.like_email) and \
                    ('\n' not in token.text) and (not token.text.strip() == ''):
                all_unigrams.append(token.text)

    return all_unigrams


def compute_overlap(ds1_texts, ds2_texts):
    dataset1_unigrams = find_unigrams(ds1_texts)
    dataset2_unigrams = find_unigrams(ds2_texts)

    set1 = set(dataset1_unigrams)
    set2 = set(dataset2_unigrams)
    common_terms = set1.intersection(set2)

    dataset1_unigrams_counter = Counter(dataset1_unigrams)
    dataset2_unigrams_counter = Counter(dataset2_unigrams)

    combined_unigrams_common_counts = {}
    for term in tqdm(common_terms):
        # Taking combined frequency count across both corpora
        combined_unigrams_common_counts[term] = dataset1_unigrams_counter[term] + \
                                                dataset2_unigrams_counter[term]

    vocab = {}
    top_100_words = list({k: v for k, v in sorted(combined_unigrams_common_counts.items(),
                                                  key=lambda item: item[1], reverse=True)}.keys())[:100]

    for term, idx in zip(top_100_words, list(range(100))):
        vocab[term] = idx

    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset1', type=str)
    parser.add_argument('dataset2', type=str)
    args = parser.parse_args()

    dataset1 = pd.read_csv(args.dataset1)
    dataset2 = pd.read_csv(args.dataset2)

    dataset1_texts = dataset1['text'].to_list()
    dataset2_texts = dataset2['text'].to_list()

    vocab = compute_overlap(dataset1_texts, dataset2_texts)

    vectorizer = CountVectorizer(vocabulary=vocab)
    dataset1_vector = vectorizer.fit_transform([' '.join(x for x in dataset1_texts)])
    dataset2_vector = vectorizer.fit_transform([' '.join(x for x in dataset2_texts)])

    res = stats.spearmanr(dataset1_vector.toarray()[0], dataset2_vector.toarray()[0])
    print(f'Spearman Correlation between {args.dataset1} and {args.dataset2}: {res.correlation}')
