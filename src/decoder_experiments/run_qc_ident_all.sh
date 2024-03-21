export CUDA_VISIBLE_DEVICES=0,4,5

python qc_identify.py --dataset 'Saptarshi7/covid_qa_cleaned_CS' --model_checkpoint 'medalpaca/medalpaca-7b'
python qc_identify.py --dataset 'Saptarshi7/covid_qa_cleaned_CS' --model_checkpoint 'tiiuae/falcon-7b-instruct'
python qc_identify.py --dataset 'Saptarshi7/covid_qa_cleaned_CS' --model_checkpoint 'garage-bAInd/Platypus2-7B'
python qc_identify.py --dataset 'Saptarshi7/covid_qa_cleaned_CS' --model_checkpoint 'google/gemma-7b-it'
python qc_identify.py --dataset 'Saptarshi7/covid_qa_cleaned_CS' --model_checkpoint 'BioMistral/BioMistral-7B'

python qc_identify.py --dataset 'squad' --model_checkpoint 'tiiuae/falcon-7b-instruct'
python qc_identify.py --dataset 'squad' --model_checkpoint 'garage-bAInd/Platypus2-7B'
python qc_identify.py --dataset 'squad' --model_checkpoint 'google/gemma-7b-it'
python qc_identify.py --dataset 'squad' --model_checkpoint 'mistralai/Mistral-7B-Instruct-v0.2'

python qc_identify.py --dataset 'ibm/duorc' --model_checkpoint 'tiiuae/falcon-7b-instruct'
python qc_identify.py --dataset 'ibm/duorc' --model_checkpoint 'garage-bAInd/Platypus2-7B'
python qc_identify.py --dataset 'ibm/duorc' --model_checkpoint 'google/gemma-7b-it'
python qc_identify.py --dataset 'ibm/duorc' --model_checkpoint 'mistralai/Mistral-7B-Instruct-v0.2'