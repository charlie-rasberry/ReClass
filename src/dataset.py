# dataset.py
# Takes a row from the csv, tokenizes the review and returns a tensor ready for the model
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class ReviewDataset(Dataset):
    """
    Dataset for tokenized reviews with labels for all 4 tasks.

    Dataset is for map style datasets like here, instead of using IteratableDataset (better for data streams).
    Expects a csv and tokenizes reviews using XLM-RoBERTa (SentencePiece), returning a dictionary with of
    input tensors and integer labels for all 4 tasks.
    """

    def __init__(self, path, tokenizer, max_length=256):
        self.df = pd.read_csv(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        review = self.df.iloc[idx]['review']

        # Tokenize with padding and truncation to max_length, returning PyTorch tensors
        encoding = self.tokenizer(review, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),

            # Labels for all 4 tasks, converted to tensors
            'bug_report': torch.tensor(self.df.iloc[idx]['bug_report'], dtype=torch.long),
            'feature_request': torch.tensor(self.df.iloc[idx]['feature_request'], dtype=torch.long),
            'aspect': torch.tensor(self.df.iloc[idx]['aspect'], dtype=torch.long),
            'aspect_sentiment': torch.tensor(self.df.iloc[idx]['aspect_sentiment'], dtype=torch.long)
        }
class InferenceDataset(Dataset):
        def __init__(self, path, tokenizer, text_column, max_length=256):
                self.df = pd.read_csv(path)
                self.tokenizer = tokenizer
                self.text_column = text_column
                self.max_length = max_length

        def __len__(self):
                return len(self.df)
        
        def __getitem__(self, idx):
                review = str(self.df.iloc[idx][self.text_column])

                if review == 'nan' or review.strip() == '':
                    review = ' '

                # Same as training dataset but without labels, for inference on test sets
                encoding = self.tokenizer(review, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
                return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                }    
    



if __name__ == "__main__":
    # Quick test
    dataset = ReviewDataset("data/processed/original_train.csv", AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base"))
    print(dataset.__getitem__(1))
    


    

    

