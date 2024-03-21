accelerate launch covidqa_ft.py --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-covidqa'


accelerate launch --mixed_precision 'fp16' --gpu_ids '7' \
EQA_ft_FDA.py --model_checkpoint 'google-bert/bert-base-uncased' --trained_model_name 'bert-base-uncased-duorc' \
--dataset 'ibm/duorc'
