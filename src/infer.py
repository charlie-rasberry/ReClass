import pandas as pd
import numpy as np
import torch
import argparse
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

# mappings
binary_map = {1:'Yes', 0:'No'}
aspect_map = {0:'App', 1:'Driver', 2:'General', 3:'Payment', 4:'Pricing', 5:'Service'}
sentiment_map = {0:'Positive', 1:'Neutral', 2:'Negative'}


SEED = 4321
torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="RECLASS, Multitask learning for review classification.")
    parser.add_argument("--model_path", type=str, help="Enter the models path / the desired .pt file")
    parser.add_argument("--task", type=str, default="all", choices=["all", "bug_report", "feature_request", "aspect", "aspect_sentiment"], help="Specific task to train for stl usage only" )
    parser.add_argument("--interactive", help="Loops reading input until exit")
    parser.add_argument("--text", help="Use command line text for input")
    return parser.parse_args()

