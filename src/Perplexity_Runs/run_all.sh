#!/bin/bash

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gres=A100:2
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="PPL"   	        # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

DATASETS=$(ls ../common_terms_freq/*.csv)

#CLM
CLMs=("tiiuae/falcon-7b-instruct"  #max_length = 2048
      "garage-bAInd/Platypus2-7B") #max_length = 4096
for ds in $DATASETS
do
   for model in "${CLMs[@]}"
   do
     python CLM_v2.py --model_checkpoint "$model" --corpus_file "$ds"
   done
done

#MLM
python MLM_ppl.py --model_checkpoint "bert-base-uncased" --corpus_file "../common_terms_freq/Saptarshi7-techqa-squad-style_validation_for_PPL_eval.csv"