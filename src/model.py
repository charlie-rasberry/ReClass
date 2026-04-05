# model.py
# Shared encoder (XLM-RoBERTa) with either multitask heads for all 4 tasks or single task head for comparison

from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
import torch.nn as nn

# Using dropout before classification to reduce overfitting
class SingleTaskModel(nn.Module):
    """Single task model with one head to compare MTL approach to review classification"""

    def __init__(self, task_name, num_classes, dropout_rate=0.2):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
        self.droput = nn.Dropout(dropout_rate)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.task_name = task_name
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output= self.droput(outputs.last_hidden_state[:, 0, :])
        logits = self.head(output)
        return {self.task_name: logits}

class Model(nn.Module): 
    """ Multitask model with shared encoder and 4 task specific heads."""
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

        hidden_size = self.encoder.config.hidden_size

        # Applied across shared cls token, before all task heads 
        self.dropout = nn.Dropout(dropout_rate)
        # get logits for each head
        self.bug_head = nn.Linear(hidden_size, 2)
        self.feature_head = nn.Linear(hidden_size, 2)
        self.aspect_head = nn.Linear(hidden_size, 6)
        self.aspect_sentiment_head = nn.Linear(hidden_size, 3)

    # Pass through encoder once then extract the token representation, then reuse the shared represenetation across all tasks
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # index 0 from [batch_size, 768]
        output = outputs.last_hidden_state[:, 0, :]

        output = self.dropout(output)

        bug_logits = self.bug_head(output)
        feature_logits = self.feature_head(output)
        aspect_logits = self.aspect_head(output)
        aspect_sentiment = self.aspect_sentiment_head(output)
        return {
            'bug_report': bug_logits,
            'feature_request': feature_logits,
            'aspect': aspect_logits,
            'aspect_sentiment': aspect_sentiment
        }
    
if __name__ == "__main__":
    from dataset import ReviewDataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    dataset = ReviewDataset("data/processed/original_train.csv", tokenizer)
    loader = DataLoader(dataset, batch_size=2)

    batch = next(iter(loader))

    model = Model()
    outputs = model(batch["input_ids"], batch["attention_mask"])

    for k, v in outputs.items():
        print(k, v.shape)