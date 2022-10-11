for trained_model in $1*/ ; do
    python noft.py --model_checkpoint $trained_model --dataset $2
done
