#!/bin/bash

python spearman_calc.py --dataset1 "squad_for_PPL_eval.csv" --dataset2 "Saptarshi7-techqa-squad-style_for_PPL_eval.csv"
python spearman_calc.py --dataset1 "squad_for_PPL_eval.csv" --dataset2 "Saptarshi7-covid_qa_cleaned_CS_for_PPL_eval.csv"
python spearman_calc.py --dataset1 "squad_for_PPL_eval.csv" --dataset2 "cuad_for_PPL_eval.csv"
python spearman_calc.py --dataset1 "squad_for_PPL_eval.csv" --dataset2 "duorc_for_PPL_eval.csv"
