BERT_VARIANT_STR=`awk -F'uncased_|.zip' '{print $2}' <<< $1`
echo "Downloading BERT variant $BERT_VARIANT_STR"

CURL='/usr/bin/curl'
RVMHTTP="$1"
CURLARGS="-O -J"

#Downloading the requested BERT variant.
$CURL $CURLARGS $RVMHTTP

#Extracting downloaded zip file.
unzip "uncased_$BERT_VARIANT_STR.zip" -d "uncased_$BERT_VARIANT_STR"

echo "Converting Tensorflow Checkpoint to PyTorch..."
export BERT_BASE_DIR="`pwd`/uncased_$BERT_VARIANT_STR"
yes | transformers-cli convert --model_type bert \
  --tf_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --config $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_output $BERT_BASE_DIR/pytorch_model.bin

#Renaming the config file.
`mv "uncased_$BERT_VARIANT_STR/bert_config.json" "uncased_$BERT_VARIANT_STR/config.json"`


# Define the key and value
KEY="model_type"
VALUE="bert"

# Define the JSON file
JSON_FILE="uncased_$BERT_VARIANT_STR/config.json"

# Use sed to add the key-value pair to the JSON file
sed -i "s/^\s*{\s*$/  \"$KEY\": \"$VALUE\",/" "$JSON_FILE"

#Removing the zip file
`rm "uncased_$BERT_VARIANT_STR.zip"`