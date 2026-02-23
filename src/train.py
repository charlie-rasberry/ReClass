# train.py
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

from dataset import ReviewDataset
from model import Model

# class weights, training loop and early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

train = "data/processed/original_train.csv"
val = "data/processed/original_val.csv"
train_dataset = ReviewDataset(train, tokenizer)
val_dataset = ReviewDataset(val, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model = Model().to(device)


# move input_ids, attention_mask and labels to device in each batch

# ------------------- Class weights -------------------
# Using weights inversely proportional to class frequencies to avoid majority class bias, 
# prioritize useful bug reports / feature requests
def compute_weights(train_df, column):
    classes = np.unique(train_df[column])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=train_df[column])
    return torch.tensor(weights, dtype=torch.float).to(device)

# -------------------- Loss functions -------------------
# just a later idea
#    1.0 * bug_loss +
#    1.0 * feature_loss +
#    0.5 * aspect_loss +
#    0.5 * sentiment_loss


# -------------------- Optimizer and scheduler -------------------




# ------------------- Training loop -------------------
# For each epoch:




# ------------------- Stopping logic -------------------
# After each epoch, find mean of 4 macro f1 scores
# If there is no improvement for 3 epochs consecutively, stop training 
# Prevents overfitting which saves time and resources




train_df = pd.read_csv(train)
bug_weights = compute_weights(train_df, 'bug_report')
feature_weights = compute_weights(train_df, 'feature_request')
aspect_weights = compute_weights(train_df, 'aspect')    
aspect_sentiment_weights = compute_weights(train_df, 'aspect_sentiment')