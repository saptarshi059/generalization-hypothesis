from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file')
    args = parser.parse_args()

    with open(args.prediction_file, 'rb') as file:
        prediction_df = pd.read_pickle(file)

    total_samples = prediction_df.shape[0]
    print(f'Total number of samples: {total_samples}')

    correctly_identified_contexts = 0
    correctly_identified_questions = 0
    correctly_identified_both = 0

    for row in tqdm(prediction_df.itertuples(index=False)):
        prediction = row.prediction.split(row.prompt)[1]
        if re.search(row.context, prediction, re.IGNORECASE):
            correctly_identified_contexts += 1
        if re.search(row.question, prediction, re.IGNORECASE):
            correctly_identified_questions += 1
        if re.search(row.context, prediction, re.IGNORECASE) and re.search(row.question, prediction, re.IGNORECASE):
            correctly_identified_both += 1

    print(f'Correctly identified Contexts: {np.round(correctly_identified_contexts/total_samples, 2)}%')
    print(f'Correctly identified Questions: {np.round(correctly_identified_questions / total_samples, 2)}%')
    print(f'Correctly identified Both: {np.round(correctly_identified_both / total_samples, 2)}%')
