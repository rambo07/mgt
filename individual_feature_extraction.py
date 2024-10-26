'''
    ---------- PLEASE READ AT ONCE --------------
1. Use model_name with its corresponding model_path 
2. individual directory -> stroing the individual in model's named directory
3. timing_csv -> for storting the time into seconds
4. data_paths -> dictionary with data_type(key) and their_path (value)
'''

import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import gc
import time  # Importing time module

# Helper function to load the datasets
def load_dataset(jsonl_file):
    with open(jsonl_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    return pd.DataFrame(data)


model_name = 'Llama-3.2-3B-Instruct'
# Load models
model_paths = [
    # "model/Llama-3.2-11B-Vision",
    # "model/Llama-3.2-11B-Vision-Instruct",
    # "model/Llama-3.2-3B",
    "model/Llama-3.2-3B-Instruct"
]

# Directory structure setup for individual features
individual_features_directory = f"timing50/{model_name}"
timing_csv = f'{individual_features_directory}/timings.csv'

# Process each dataset and extract features for both human (label0) and machine (label1)
data_paths = {
  "train": "half_data/en_train50.jsonl",  
    "dev": "half_data/en_dev50.jsonl"
}

# Incrementing counters for filenames
file_counters = {
    "train": 2,  # Starting number for 'train' files
    "dev": 2     # Starting number for 'dev' files
}


os.makedirs(individual_features_directory, exist_ok=True)

# Function to extract features
def extract_features(model, tokenizer, texts, batch_size=16):  
    model.eval()
    features = []

    # Ensure model uses DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features", unit="batch"):
        batch_texts = texts[i:i + batch_size]

        # Tokenize and move to GPU
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

        with torch.no_grad():
            # Use autocast for mixed precision
            with torch.autocast(device_type='cuda', enabled=True):
                outputs = model(**tokens)

        logits = outputs.logits
        predicted_probs = torch.softmax(logits, dim=-1)

        # Adjust for your needs
        predicted_probs = predicted_probs[:, :-1, :]
        log_prob_predicted = torch.max(predicted_probs, dim=-1).values
        log_prob_observed = predicted_probs.gather(2, tokens['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
        entropy = -torch.sum(predicted_probs * torch.log(predicted_probs + 1e-9), dim=-1)

        # Convert to CPU and numpy
        log_prob_predicted = log_prob_predicted.cpu().numpy()
        log_prob_observed = log_prob_observed.cpu().numpy()
        entropy = entropy.cpu().numpy()

        # Check for empty feature batches
        if log_prob_predicted.shape[0] == 0 or log_prob_observed.shape[0] == 0 or entropy.shape[0] == 0:
            continue

        # Align lengths and concatenate
        min_length = min(log_prob_predicted.shape[1], log_prob_observed.shape[1], entropy.shape[1])
        log_prob_predicted = log_prob_predicted[:, :min_length]
        log_prob_observed = log_prob_observed[:, :min_length]
        entropy = entropy[:, :min_length]

        # Stack features and append to list
        batch_features = np.array([log_prob_predicted, log_prob_observed, entropy]).transpose(1, 2, 0)
        features.append(batch_features)

        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

    # Ensure non-empty features and concatenate them
    if len(features) > 0:
        return np.concatenate(features, axis=0)
    else:
        return np.array([])  # Return empty array if no features were extracted

# # Load models
# model_paths = [
#     # "model/Llama-3.2-11B-Vision",
#     # "model/Llama-3.2-11B-Vision-Instruct",
#     # "model/Llama-3.2-3B",
#     "model/Llama-3.2-3B-Instruct"
# ]
models = {}
for model_path in model_paths:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    # model.half()  # Use half precision
    model_name = os.path.basename(model_path)  # Get model name from path
    models[model_name] = (model, tokenizer)  # Use the model name as the key

# # Incrementing counters for filenames
# file_counters = {
#     "train": 2,  # Starting number for 'train' files
#     "dev": 2     # Starting number for 'dev' files
# }



# List to store time taken and number of rows processed for each extraction
timing_info = []

# Process each dataset and extract features for both human (label0) and machine (label1)
# data_paths = {
#   "train": "half_data/en_train50.jsonl",  
#     "dev": "half_data/en_dev50.jsonl"
# }

for dataset_type, jsonl_path in data_paths.items():
    print(f"Processing {dataset_type} dataset...")

    # Load dataset
    df = load_dataset(jsonl_path)

    # Test with only the first 50 rows for each dataset
    # df = df.head(50)  # Limit to 50 rows

    # Loop through both labels: human (label0) and machine (label1)
    for label_type, label_value in [("human", 0), ("machine", 1)]:
        filtered_df = df[df['label'] == label_value]  # Filter by label (0 = human, 1 = machine)

        # Track the number of rows for this label type
        num_rows = len(filtered_df)

        # Convert to Hugging Face Dataset
        dataset_filtered = Dataset.from_pandas(filtered_df)

        # Extract texts
        texts = [entry["text"] for entry in dataset_filtered]

        if len(texts) == 0:
            print(f"No data available for label '{label_type}' in {dataset_type} dataset. Skipping...")
            continue

        # Extract and save features for each model
        for model_name, (model, tokenizer) in models.items():
            # Start timing
            start_time = time.time()  
            features = extract_features(model, tokenizer, texts, batch_size=16)
            elapsed_time = time.time() - start_time  # Calculate elapsed time

            if features.size == 0:
                print(f"No features extracted for {label_type} in {dataset_type}. Skipping...")
                continue

            # Create filename in the desired format, using the appropriate counter for 'train' or 'dev'
            label = "label0" if label_value == 0 else "label1"
            filename = f"{model_name}_features({label}_{dataset_type}{file_counters[dataset_type]}).npy"
            file_counters[dataset_type] += 1  # Increment the counter for the respective dataset

            # Save the features in the individual feature directory
            save_path = os.path.join(individual_features_directory, dataset_type, label_type)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, filename), features)
            print(f"Features saved to: {os.path.join(save_path, filename)}")

            # Record timing information and number of rows processed
            timing_info.append({
                "model_name": model_name,
                "label_type": label_type,
                "dataset_type": dataset_type,
                "elapsed_time": elapsed_time,
                "num_rows": num_rows,  # Number of rows processed
                # "file_path": os.path.join(save_path, filename)
            })

# Save timing information to a CSV file, including the number of rows
timing_df = pd.DataFrame(timing_info)
timing_df.to_csv(timing_csv, index=False)

print("Feature extraction and saving completed!")
