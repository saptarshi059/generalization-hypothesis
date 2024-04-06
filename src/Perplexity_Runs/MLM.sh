#MLM
DATASETS=$(ls ../common_terms_freq/*.csv)
MLMs=("roberta-base"
      "bert-base-uncased")
for ds in $DATASETS
do
   for model in "${MLMs[@]}"
   do
     python MLM_ppl.py --model_checkpoint "$model" --corpus_file "$ds"
   done
done