#!/bin/bash

:'
python SS_tests.py --dataset sense_data-covidqa.csv --model_name dmis-lab/biobert-base-cased-v1.2
python SS_tests.py --dataset sense_data-covidqa.csv --model_name allenai/scibert_scivocab_uncased
python SS_tests.py --dataset sense_data-cuad.csv --model_name ProsusAI/finbert
python SS_tests.py --dataset sense_data-cuad.csv --model_name nlpaueb/legal-bert-base-uncased
python SS_tests.py --dataset sense_data-techqa.csv --model_name allenai/scibert_scivocab_uncased
'

MODELS=('tiiuae/falcon-7b-instruct'
        'garage-bAInd/Platypus2-7B'
        'google/gemma-7b-it'
        'mistralai/Mistral-7B-Instruct-v0.2')

DATASETS=$(ls ../../data/sense_data/*.csv)

for ds in $DATASETS
do
   for model in "${MODELS[@]}"
   do
     python SS_tests.py --dataset "$ds" --model_name "$model"
   done
done
