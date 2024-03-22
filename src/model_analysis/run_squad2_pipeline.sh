#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

BERTS=("https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-512_A-8.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-768_A-12.zip"

                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-768_A-12.zip"

                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-512_A-8.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-6_H-768_A-12.zip"

                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-768_A-12.zip"

                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-512_A-8.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-10_H-768_A-12.zip"

                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-128_A-2.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-256_A-4.zip"
                "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-512_A-8.zip"
                )

DATASETS=('Saptarshi7/techqa-squad-style' 'ibm/duorc' 'cuad')

for current_dataset in "${DATASETS[@]}"
do
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Current Dataset: $current_dataset<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
  for model in "${BERTS[@]}"
  do
     . ./BERT_downloader.sh "$model"
     accelerate launch --mixed_precision 'fp16' --gpu_ids '7' squad_ft.py --squad_version2 True \
     --model_checkpoint "uncased_$BERT_VARIANT_STR" \
     --trained_model_name "uncased_${BERT_VARIANT_STR}_squad2"
    python "../answer_length_analysis/noft.py" --model_checkpoint "uncased_${BERT_VARIANT_STR}_squad2" --dataset "$current_dataset"
  done
done
