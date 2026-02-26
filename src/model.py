# model.py
# One encoder, four shared heads(bug report, feature request, aspect, aspect sentiment)
# 12 transformer layers, 12 attention heads

from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
import torch.nn as nn

# Using dropout, This has proven to be an effective technique 
# for regularization and preventing the co-adaptation of neurons as described in https://arxiv.org/abs/1207.0580

# Each nn.linear is used to map RoBERTa's hidden representation onto the output space of each task head
# Each hidden representation is size 768

class SingleTaskModel(nn.Module): #   TASK-SPECIFIC/SINGLE-TASK MODEL ARCHITECTURE
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

class Model(nn.Module): #   MULTITASK MODEL ARCHITECTURE
    def __init__(self, dropout_rate=0.2): # Try other p values
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")

        hidden_size = self.encoder.config.hidden_size

        # Applied across whole output, shared
        self.dropout = nn.Dropout(dropout_rate)

        self.bug_head = nn.Linear(hidden_size, 2)
        self.feature_head = nn.Linear(hidden_size, 2)
        self.aspect_head = nn.Linear(hidden_size, 6)
        self.aspect_sentiment_head = nn.Linear(hidden_size, 3)

    # Pass through encoder then extract the token representation
    # Apply droupout to it, take scores for each head, return them in a dictionary
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = outputs.last_hidden_state[:, 0, :]

        output = self.dropout(output)

        # Logits for each head:
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