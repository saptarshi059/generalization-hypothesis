export CUDA_VISIBLE_DEVICES=1
export SEED_ID=42
export DATA_DIR="../../data/json_data/squad/"
export MODEL_TYPE='bert'
export MODEL_NAME_OR_PATH='bert-base-uncased'
export VERSION_2_WITH_NEGATIVE=False
export CACHE_DIR="../../data/json_data/squad/"
export OUTPUT_DIR="../../data/json_data/squad/"

python ./run_textemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --overwrite_output_dir \
    --do_lower_case \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --train_file ${DATA_DIR}/train.json \
    --max_seq_length 384 \
    --per_gpu_train_batch_size=12  \
    --num_train_epochs 1 \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE_DIR} \
    --seed ${SEED_ID}