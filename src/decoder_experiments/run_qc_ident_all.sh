#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=1
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="Decoder-Correctly_Ident" # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

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