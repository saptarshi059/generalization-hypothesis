#!/bin/sh

files=$(ls ./*pkl)

for prediction_file in $files;
do
  echo "$prediction_file"
  formatted_file_name=$(echo "$prediction_file" | sed 's/\.\///')
  python qc_ident_eval.py --prediction_file "$formatted_file_name"
done