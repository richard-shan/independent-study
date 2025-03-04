from tmnt.estimator import SeqBowEstimator
import numpy as np
import os
import logging
import torch
import pandas as pd
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.distribution import LogisticGaussianDistribution
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import get_llm_dataloader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils import shuffle

# Load the new dataset
file_path = 'C:/independentStudy/independent-study/testing/synthetic.csv'
data = pd.read_csv(file_path)

# Extract relevant columns
queries = data['questions'].tolist()
ground_truths = data['answers'].tolist()
responses = data['responses'].tolist()
labels = data['binary_correctness'].astype(int).tolist()

# Combine query, ground truth, and response for input
combined_inputs = [f"Query: {q} Ground Truth: {gt} Response: {r}" for q, gt, r in zip(queries, ground_truths, responses)]

# Vectorizer
vectorizer = TMNTVectorizer(vocab_size=2000, count_vectorizer_kwargs={'token_pattern': r'\b[A-Za-z][A-Za-z][A-Za-z]+\b'})
X, _ = vectorizer.fit_transform(combined_inputs)

# Logging Configuration
use_logging = True
if use_logging:
    logging_config(folder='.', name='f_seqbow_binary_classification', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

# DataLoader Preparation
tf_llm_name = 'distilbert-base-uncased'
batch_size = 16
seq_len = 256

# Prepare datasets for dataloaders
train_size = int(0.8 * len(data))  # 80% training data, 20% validation data
train_ds = list(zip(labels[:train_size], combined_inputs[:train_size]))
dev_ds = list(zip(labels[train_size:], combined_inputs[train_size:]))

label_map = {0: 0, 1: 1}

device_str = 'cuda:2' if torch.cuda.is_available() else 'cpu'
train_loader = get_llm_dataloader(train_ds, vectorizer, tf_llm_name, label_map, batch_size, seq_len, device=device_str)
dev_loader = get_llm_dataloader(dev_ds, vectorizer, tf_llm_name, label_map, batch_size, seq_len, device=device_str)

# Model Initialization
num_topics = 20  # Keeping it same, but it's not used for binary classification
latent_distribution = LogisticGaussianDistribution(768, num_topics, dr=0.1, alpha=2.0, device=device_str)
device = torch.device(device_str)

estimator = SeqBowEstimator(
    llm_model_name=tf_llm_name,
    latent_distribution=latent_distribution,
    n_labels=2,  # Binary classification
    vocabulary=vectorizer.get_vocab(),
    batch_size=batch_size,
    device=device,
    log_interval=1,
    log_method=log_method,
    gamma=100.0,
    lr=2e-5,
    decoder_lr=0.01,
    epochs=4
)

# Train and save the model
estimator.fit_with_validation(train_loader, dev_loader, aux_data=None)

os.makedirs('_model_dir', exist_ok=True)
estimator.write_model('./_model_dir')

# Print training and validation accuracy separately
train_accuracy = estimator.validate(estimator.model, train_loader)
val_accuracy = estimator.validate(estimator.model, dev_loader)
print(f'Training Accuracy: {train_accuracy}')
print(f'Validation Accuracy: {val_accuracy}')

# Inference Object
inferencer = SeqVEDInferencer(estimator, max_length=seq_len, pre_vectorizer=vectorizer)

# Example inference
test_query = "What is the largest planet in our solar system?"
test_ground_truth = "Jupiter"
test_response = "Jupiter is the largest planet in our solar system."
test_input = f"Query: {test_query} Ground Truth: {test_ground_truth} Response: {test_response}"

# Transform and predict
token_result = inferencer.vectorizer.transform([test_input])

# Debug print to check token_result
print(token_result)

# Check if attention mask is None and handle it
if token_result[1] is None:
    print("Attention mask is not generated.")
else:
    # Convert the sparse matrix to dense tensor and move to the device
    input_ids = torch.tensor(token_result[0].todense()).to(device)
    attention_mask = torch.tensor(token_result[1].todense()).to(device)

    # Perform the prediction
    predictions = inferencer.model.forward_encode(input_ids, attention_mask)
    print(predictions)  # Should print the classification of the response as "correct" or "incorrect"

