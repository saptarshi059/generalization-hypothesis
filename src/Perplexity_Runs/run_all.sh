#!/bin/bash

DATASETS=$(ls ../common_terms_freq/*.csv)
CLMs=("tiiuae/falcon-7b-instruct"
      "garage-bAInd/Platypus2-7B"
      "google/gemma-7b-it"
      "mistralai/Mistral-7B-Instruct-v0.2")

#CLM
for ds in $DATASETS
do
   for model in "${CLMs[@]}"
   do
     python CLM_ppl.py --model_checkpoint "$model" --corpus_file "$ds"
   done
done

#MLM
for ds in $DATASETS
do
  python MLM_ppl.py --model_checkpoint "roberta-base" --corpus_file "$ds"
done