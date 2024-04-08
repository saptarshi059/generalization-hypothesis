:'
python SS_tests.py --dataset sense_data-covidqa.csv --model_name dmis-lab/biobert-base-cased-v1.2
python SS_tests.py --dataset sense_data-covidqa.csv --model_name allenai/scibert_scivocab_uncased
python SS_tests.py --dataset sense_data-cuad.csv --model_name ProsusAI/finbert
python SS_tests.py --dataset sense_data-cuad.csv --model_name nlpaueb/legal-bert-base-uncased
python SS_tests.py --dataset sense_data-techqa.csv --model_name allenai/scibert_scivocab_uncased
'

MODELS=('Falcon'
        'Platypus'
        'Gemma'
        'Mistral')

DATASETS=$(ls ../data/sense_data/*.csv)

for ds in $DATASETS
do
   for model in "${MODELS[@]}"
   do
     python SS_tests --dataset "$ds" --model_name "$model"
   done
done