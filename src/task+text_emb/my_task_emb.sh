#!/bin/bash

JSON_DATASET_PATHS=("../../data/json_data/squad/task_emb"
                    "../../data/json_data/covidqa/task_emb"
                    "../../data/json_data/techqa/task_emb"
                    "../../data/json_data/duorc/task_emb"
                    "../../data/json_data/cuad/task_emb")

for dataset_path in "${JSON_DATASET_PATHS[@]}"
do
  export CUDA_VISIBLE_DEVICES=1
  export SEED_ID=42
  export DATA_DIR=$dataset_path
  export MODEL_TYPE="bert"
  export USE_LABELS=True # set to False to sample from the model's predictive distribution
  export CACHE_DIR=$dataset_path
  export MODEL_NAME_OR_PATH="bert-base-uncased-squad-v1"
  # we start from a fine-tuned task-specific BERT so no need for further fine-tuning
  export FURTHER_FINETUNE_CLASSIFIER="False"
  export FURTHER_FINETUNE_FEATURE_EXTRACTOR="False"
  export OUTPUT_DIR=$dataset_path

  if [[ $dataset_path == *"squad"* ]] || [[ $dataset_path == *"covidqa"* ]]; then
    negative_flag='False'
  else
    negative_flag='True'
  fi
  export VERSION_2_WITH_NEGATIVE=$negative_flag

  python ./run_taskemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --use_labels ${USE_LABELS} \
    --do_lower_case \
    --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
    --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
    --train_file "${DATA_DIR}"/train.json \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --max_seq_length 384 \
    --batch_size=12  \
    --learning_rate 3e-5 \
    --num_epochs 1 \
    --doc_stride 128 \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --seed "${SEED_ID}" \
    --overwrite_output_dir
done