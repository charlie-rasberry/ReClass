# dataset.py

# Takes a row from the csv, tokenizes the review and returns a tensor
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ReviewDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=256):
        self.df = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        review = self.df.iloc[idx]['review']
        return review

uber = ReviewDataset("data/processed/original_train.csv", AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base"))
print(uber.__getitem__(1))

    

    

