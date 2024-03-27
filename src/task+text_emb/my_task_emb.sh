export SEED_ID=42
export DATA_DIR="../../data/json_data/squad"
export MODEL_TYPE="bert"
export VERSION_2_WITH_NEGATIVE="False"
export USE_LABELS=True # set to False to sample from the model's predictive distribution
export CACHE_DIR="../../data/json_data/squad"
export MODEL_NAME_OR_PATH="deepset/bert-base-uncased-squad2"
# we start from a fine-tuned task-specific BERT so no need for further fine-tuning
export FURTHER_FINETUNE_CLASSIFIER=False
export FURTHER_FINETUNE_FEATURE_EXTRACTOR=False
export OUTPUT_DIR="../../data/json_data/squad"

python ./run_taskemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --use_labels ${USE_LABELS} \
    --do_lower_case \
    --finetune_classifier ${FURTHER_FINETUNE_CLASSIFIER} \
    --finetune_feature_extractor ${FURTHER_FINETUNE_FEATURE_EXTRACTOR} \
    --train_file ${DATA_DIR}/train.json \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --max_seq_length 384 \
    --batch_size=12  \
    --learning_rate 3e-5 \
    --num_epochs 1 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID} \
    --overwrite_output_dir