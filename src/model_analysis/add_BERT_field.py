import json
import sys

# Define the key and value
key = "model_type"
value = "bert"

# Define the path to the JSON file
json_file_path = sys.argv[1]

# Read the existing JSON data
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# Add the new key-value pair
data[key] = value

# Write the updated JSON data back to the file
with open(json_file_path, 'w') as json_file:
    json.dump(data, json_file, indent=4)  # indent for pretty printing
