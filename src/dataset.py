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

        # encoding['input_ids'] 1D tensor of token ids, shape [max_length]
        # encoding['attention_mask'] 1D tensor of 1s 0s showing real tokens vs padding, shape [max_length]
        # Both have shape [1, max_length] because of return_tensors='pt'
        # Squeeze them to [max_length] with .squeeze(0)
        encoding = self.tokenizer(
                review,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Returns a dictionary with:
        #   'input_ids': tensor of shape [max_length]
        
        #   'attention_mask': tensor of shape [max_length]

        # MTL structure labels as tensor scalars:
        #   'bug_report': tensor scalar (torch.tensor(label_value))
        #   'feature_request': tensor scalar (torch.tensor(label_value))
        #   'aspect': tensor scalar (torch.tensor(label_value))
        #   'aspect_sentiment': tensor scalar (torch.tensor(label_value))
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bug_report': torch.tensor(self.df.iloc[idx]['bug_report']),
            'feature_request': torch.tensor(self.df.iloc[idx]['feature_request']),
            'aspect': torch.tensor(self.df.iloc[idx]['aspect']),
            'aspect_sentiment': torch.tensor(self.df.iloc[idx]['aspect_sentiment'])
        }

# uber = ReviewDataset("data/processed/original_train.csv", AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base"))
# print(uber.__getitem__(1))

    

    

