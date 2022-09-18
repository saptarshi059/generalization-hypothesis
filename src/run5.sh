export CUDA_VISIBLE_DEVICES=5

python predictions_gen.py --model_checkpoint rc-bidaf --dataset Saptarshi7/covid_qa_cleaned_CS
python predictions_gen.py --model_checkpoint rc-naqanet --dataset Saptarshi7/covid_qa_cleaned_CS
python noft.py --model_checkpoint csarron/bert-base-uncased-squad-v1 --dataset Saptarshi7/covid_qa_cleaned_CS
python noft.py --model_checkpoint csarron/roberta-base-squad-v1 --dataset Saptarshi7/covid_qa_cleaned_CS

python predictions_gen.py --model_checkpoint rc-naqanet --dataset cuad
python noft.py --model_checkpoint csarron/bert-base-uncased-squad-v1 --dataset cuad
python noft.py --model_checkpoint csarron/roberta-base-squad-v1 --dataset cuad
