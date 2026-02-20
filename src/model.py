# model.py
# One encoder, four shared heads(bug report, feature request, aspect, aspect sentiment)
# 12 transformer layers, 12 attention heads

from transformers import AutoTokenizer, AutoModelForMaskedLM, XLMRobertaModel
import torch.nn as nn

# Using dropout, This has proven to be an effective technique 
# for regularization and preventing the co-adaptation of neurons as described in https://arxiv.org/abs/1207.0580

# Each nn.linear is used to map RoBERTa's hidden representation onto the output space of each task head
# Each hidden representation is size 768
class Model(nn.Module):
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
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)






tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("FacebookAI/xlm-roberta-base")