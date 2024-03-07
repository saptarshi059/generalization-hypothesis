export CUDA_VISIBLE_DEVICES=1

#python benchmarking.py --model_checkpoint google/gemma-7b-it --dataset Saptarshi7/covid_qa_cleaned_CS
#python benchmarking.py --model_checkpoint BioMistral/BioMistral-7B --dataset Saptarshi7/covid_qa_cleaned_CS

python benchmarking.py --model_checkpoint tiiuae/falcon-7b-instruct --dataset squad
python benchmarking.py --model_checkpoint garage-bAInd/Platypus2-7B --dataset squad
python benchmarking.py --model_checkpoint google/gemma-7b-it --dataset squad
python benchmarking.py --model_checkpoint mistralai/Mistral-7B-Instruct-v0.2 --dataset squad

python benchmarking.py --model_checkpoint tiiuae/falcon-7b-instruct --dataset ibm/duorc
python benchmarking.py --model_checkpoint garage-bAInd/Platypus2-7B --dataset ibm/duorc
python benchmarking.py --model_checkpoint google/gemma-7b-it --dataset ibm/duorc
python benchmarking.py --model_checkpoint mistralai/Mistral-7B-Instruct-v0.2 --dataset ibm/duorc

python benchmarking.py --model_checkpoint tiiuae/falcon-7b-instruct --dataset Saptarshi7/techqa-squad-style
python benchmarking.py --model_checkpoint garage-bAInd/Platypus2-7B --dataset Saptarshi7/techqa-squad-style
python benchmarking.py --model_checkpoint google/gemma-7b-it --dataset Saptarshi7/techqa-squad-style
python benchmarking.py --model_checkpoint mistralai/Mistral-7B-Instruct-v0.2 --dataset Saptarshi7/techqa-squad-style
python benchmarking.py --model_checkpoint microsoft/phi-2 --dataset Saptarshi7/techqa-squad-style

python benchmarking.py --model_checkpoint tiiuae/falcon-7b-instruct --dataset cuad
python benchmarking.py --model_checkpoint garage-bAInd/Platypus2-7B --dataset cuad
python benchmarking.py --model_checkpoint google/gemma-7b-it --dataset cuad
python benchmarking.py --model_checkpoint mistralai/Mistral-7B-Instruct-v0.2 --dataset cuad
python benchmarking.py --model_checkpoint AdaptLLM/law-LLM --dataset cuad
