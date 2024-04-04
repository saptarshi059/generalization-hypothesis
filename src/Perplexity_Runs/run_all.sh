#!/bin/bash

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="PPL"   	        # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

DATASETS=$(ls ../common_terms_freq/*.csv)
CLMs=("tiiuae/falcon-7b-instruct"
      "garage-bAInd/Platypus2-7B"
      "google/gemma-7b-it"
      "mistralai/Mistral-7B-Instruct-v0.2")

#CLM
for ds in $DATASETS
do
   for model in "${CLMs[@]}"
   do
     python CLM_ppl.py --model_checkpoint "$model" --corpus_file "$ds"
   done
done

#MLM
for ds in $DATASETS
do
  python MLM_ppl.py --model_checkpoint "roberta-base" --corpus_file "$ds"
done