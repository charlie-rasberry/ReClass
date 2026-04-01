# infer.py
from datetime import datetime
import os
import torch
import time
import argparse
import json
import torch.nn.functional as F
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset


from dataset import InferenceDataset
from model import Model, SingleTaskModel



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
    parser.add_argument("--model_path", type=str, required=True, help=".pt file in outputs/")
    parser.add_argument("--task", type=str, default="all", choices=["all", "bug_report", "feature_request", "aspect", "aspect_sentiment"])
    parser.add_argument("--interactive", action="store_true", help="Loops reading input until exit()")
    parser.add_argument("--text", action="store_true", help="Use command line text for input")
    parser.add_argument("--dataset", type=str, help="Enter a file name for inference (stored in data/processed/)", default="review")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mode", type=str, required=True, choices=["mtl", "stl"], help="mtl or stl")
    parser.add_argument("--text_column", type=str, default="review", help="Where is the text column")

    return parser.parse_args()



def main():
    os.makedirs("outputs/inference", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # this section is nearly identical to the first part of evaluate.py
    args = parse_args()
    print(f'{"="*50}')
    print(f'{"Starting inference"}')
    if torch.cuda.is_available():
        print(f"Using CUDA for inference: {torch.cuda.get_device_name(0)}")
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.manual_seed(SEED)
    else:
        print(f'{" "*15, "No GPUs available"}')
    print(f'{"="*50}\n')
    print(f"Running inference on: outputs/{args.model_path} using data/processed/{args.dataset}.csv")
    print("Loading model, tokenizer and datasets ...")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    # Let the user decide if they want to run inference on the whole dataset or via the shell input
    if not args.interactive and not args.text:
        infer = f"data/processed/{args.dataset}.csv"
        infer_df = pd.read_csv(infer)
        filename = f"outputs/inference/{args.model_path}_{args.task}_predictions_{args.dataset}.csv"
    else:
        infer_df = pd.DataFrame(columns=[args.text_column])
        print("Entering interactive mode. Type 'exit()' to quit.")
        while True:
            user_input = input("Enter text for inference: ")
            if user_input.lower() == "exit()":
                break
            infer_df = pd.concat([infer_df, pd.DataFrame({args.text_column: [user_input]})], ignore_index=True)
            filename = f"outputs/inference/{args.model_path}_{args.task}_predictions_interactive.csv"
            infer_df.to_csv(filename, index=False)
            infer = filename
            
    if infer is not None:
        infer_data = InferenceDataset(infer, tokenizer, args.text_column)
        infer_loader = DataLoader(infer_data, batch_size=args.batch_size)
    else:
        print("No dataset provided for inference. Exiting.")
        return

    if args.mode == "mtl":
        model = Model().to(device)
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(f"outputs/{args.model_path}", map_location=device))
        model.eval()
        active_tasks = ['bug_report', 'feature_request', 'aspect', 'aspect_sentiment']
    else: 
        if args.task == "all":
            raise ValueError("For STL, please specify a single task with --task")
        task_classes = {
            'bug_report': 2,
            'feature_request': 2,
            'aspect': 6,
            'aspect_sentiment': 3
            }
        model = SingleTaskModel(args.task, task_classes[args.task]).to(device)
        active_tasks = [args.task]
        print(f"Loading weights from {args.model_path}...")
        model.load_state_dict(torch.load(f"outputs/{args.model_path}", map_location=device))
        model.eval()

    all_preds =       {task: [] for task in active_tasks}
    all_confidences = {task: [] for task in active_tasks}
    print(f"Running inference on {args.dataset} dataset")
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    with torch.no_grad():
        for batch in infer_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            for task in active_tasks:
                logits = outputs[task]
                preds = torch.argmax(logits, dim=1)

                probs = F.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values

                all_preds[task].extend(preds.cpu().numpy())
                all_confidences[task].extend(confidence.cpu().numpy())

    end_time = time.time()
    df = pd.DataFrame({"text": infer_df[args.text_column]})

    for task in active_tasks:  # ensures ALL tasks included
        df[f"{task}_pred"] = [label_names[task][p] for p in all_preds[task]]
        df[f"{task}_confidence"] = all_confidences[task]

    output_path = filename
    df.to_csv(output_path, index=False)
    
    if not args.text:
        print(f"Inference finished. Predictions saved to {output_path}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    else:
        print(f"Inference completed in {end_time - start_time:.2f} seconds.\n")
        print(df.to_string(index=False))
        again = input("Do you want to enter another text for inference? (y/n): ")
        if again.lower() == 'y':
            main()
        else:
            print("Exiting interactive inference.")

    

if __name__ == "__main__":
    main()