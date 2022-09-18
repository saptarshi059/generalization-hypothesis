export CUDA_VISIBLE_DEVICES=4

python predictions_gen.py --model_checkpoint rc-bidaf --dataset Saptarshi7/techqa-squad-style
python predictions_gen.py --model_checkpoint rc-naqanet --dataset Saptarshi7/techqa-squad-style
python noft.py --model_checkpoint phiyodr/bert-base-finetuned-squad2 --dataset Saptarshi7/techqa-squad-style
python noft.py --model_checkpoint navteca/roberta-base-squad2 --dataset Saptarshi7/techqa-squad-style
