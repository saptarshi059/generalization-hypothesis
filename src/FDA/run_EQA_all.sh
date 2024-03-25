#!/bin/sh

#SBATCH --nodes=1                   # Number of nodes to request
#SBATCH --gpus=1                    # Number of GPUs to request
#SBATCH --cpus-per-task=4           # Number of CPUs per node to request
#SBATCH --job-name="FDA"   	        # A nice readable name of your job, to see it in the queue, instead of numbers
#SBATCH --output=jobName.%J.out     # Store the output console text to a file called jobName.<assigned job number>.out
#SBATCH --error=jobName.%J.err      # Store the error messages to a file called jobName.<assigned job number>.err
#SBATCH --mail-type=FAIL,BEGIN,END  # Send an email when the job starts, ends, or fails
#SBATCH --mail-user=saptarshi.sengupta@l3s.de # Email address to send the email to

accelerate launch covidqa_ft.py --mixed_precision 'fp16' --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-covidqa'

accelerate launch --mixed_precision 'fp16' EQA_ft_FDA.py --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-techqa' --dataset 'Saptarshi7/techqa-squad-style' --impossible_questions True

accelerate launch --mixed_precision 'fp16' EQA_ft_FDA.py --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-cuad' --dataset 'cuad' --impossible_questions True

accelerate launch --mixed_precision 'fp16' EQA_ft_FDA.py --model_checkpoint 'google-bert/bert-base-uncased' \
--trained_model_name 'bert-base-uncased-duorc' --dataset 'Saptarshi7/duorc_processed' --impossible_questions True
