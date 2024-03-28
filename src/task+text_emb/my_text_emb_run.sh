JSON_DATASET_PATHS=("../../data/json_data/squad"
                    "../../data/json_data/covidqa"
                    "../../data/json_data/techqa"
                    "../../data/json_data/duorc"
                    "../../data/json_data/cuad")

for dataset_path in "${JSON_DATASET_PATHS[@]}"
do
  export CUDA_VISIBLE_DEVICES=0
  export SEED_ID=42
  export DATA_DIR=$dataset_path
  export CACHE_DIR=$dataset_path
  export OUTPUT_DIR=$dataset_path
  export MODEL_TYPE='bert'
  export MODEL_NAME_OR_PATH='bert-base-uncased'

  if [[ $dataset_path == *"squad"* ]] || [[ $dataset_path == *"covidqa"* ]]; then
    negative_flag='False'
  else
    negative_flag='True'
  fi
  export VERSION_2_WITH_NEGATIVE=$negative_flag

  python ./run_textemb_QA.py \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --do_lower_case \
    --version_2_with_negative ${VERSION_2_WITH_NEGATIVE} \
    --train_file "${DATA_DIR}"/train.json \
    --max_seq_length 384 \
    --per_gpu_train_batch_size=12  \
    --num_train_epochs 1 \
    --doc_stride 128 \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --seed "${SEED_ID}" \
    --overwrite_output_dir
done