#train.py

from transformers import AutoTokenizer


class multiTaskModel():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")