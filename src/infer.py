import pandas as pd
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

# mappings
binary_map = {1:'Yes', 0:'No'}
aspect_map = {0:'App', 1:'Driver', 2:'General', 3:'Payment', 4:'Pricing', 5:'Service'}
sentiment_map = {0:'Positive', 1:'Neutral', 2:'Negative'}

label_names = {
    'bug_report': ['No', 'Yes'],
    'feature_request': ['No', 'Yes'],
    'aspect': ['App', 'Driver', 'General', 'Payment', 'Pricing', 'Service'],
    'aspect_sentiment': ['Positive', 'Neutral', 'Negative']
}

SEED = 4321
torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="RECLASS, Multitask learning for review classification.")
    parser.add_argument("--model_path", type=str, required=True, help=".pt file path")
    parser.add_argument("--task", type=str, default="all", choices=["all", "bug_report", "feature_request", "aspect", "aspect_sentiment"])
    parser.add_argument("--interactive", help="Loops reading input until exit()")
    parser.add_argument("--text", help="Use command line text for input")
    parser.add_argument("--dataset", type=str, required=True, help="Enter a file for inference")

    return parser.parse_args()



def main():
    args = parse_args()
    print(f'='*50)
    print(f' '*15 + "Starting inference")
    if torch.cuda.is_available():
        print(f' '*15 + "GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.manual_seed(SEED)
    else:
        print(f' '*15 + "No GPUs available")
    print(f'='*50 + "\n")
    print(f"Running inference on: {args.model_path.upper()} using {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    infer_data = f"data/processed/{args.dataset}_infer.csv"

if __name__ == main():
    main()