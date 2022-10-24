for prediction_file in $1*/ ; do
    echo $prediction_file
    python eval.py --pred_file $prediction_file --metric $2
done
