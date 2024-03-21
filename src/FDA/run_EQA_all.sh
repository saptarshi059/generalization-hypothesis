accelerate launch covidqa_ft.py --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-covidqa'


accelerate launch EQA_ft_FDA.py --model_checkpoint 'google-bert/bert-base-uncased' --trained_model_name biobert-squad --random_state $seed