# evauluate.py
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

from dataset import ReviewDataset
from model import Model, SingleTaskModel

# TODO: load checkpoint, produce tables of evaluation figures
SEED = 4321
torch.manual_seed(SEED)
np.random.seed(SEED)

# Label names for classification report, readable format instead of numeric
label_names = {
    'bug_report': ['No', 'Yes'],
    'feature_request': ['No', 'Yes'],
    'aspect': ['App', 'Driver', 'General', 'Payment', 'Pricing', 'Service'],
    'aspect_sentiment': ['Positive', 'Neutral', 'Negative']
}

def parse_args():
    parser = argparse.ArgumentParser(description="RECLASS Evaluation Script")
    parser.add_argument("--mode", type=str, required=True, choices=["mtl", "stl"], help="mtl or stl")
    parser.add_argument("--task", type=str, default="all", choices=["all", "bug_report", "feature_request", "aspect", "aspect_sentiment"])
    parser.add_argument("--dataset", type=str, required=True, choices=["original", "boosted"])
    parser.add_argument("--model_path", type=str, required=True, help=".pt file path")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Evaluating {args.mode.upper()} model on {args.dataset} dataset for task: {args.task}")

    os.makedirs("outputs/figures", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test = f"data/processed/{args.dataset}_test.csv"
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    test_dataset = ReviewDataset(test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    if args.mode == "mtl":
        model = Model().to(device)
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
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_labels =      {task: [] for task in active_tasks}
    all_preds =       {task: [] for task in active_tasks}
    all_confidences = {task: [] for task in active_tasks}

    print("Running inference on test set")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            for task in active_tasks:
                labels = batch[task].to(device)
                logits = outputs[task]
                preds = torch.argmax(logits, dim=1)

                probs = F.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values

                all_labels[task].extend(labels.cpu().numpy())
                all_preds[task].extend(preds.cpu().numpy())
                all_confidences[task].extend(confidence.cpu().numpy())
    
    summary = {
        "mode": args.mode,
        "dataset": args.dataset,
        "task": args.task,
        "model_path": args.model_path,
        "results": {}
    }

    test_df = pd.read_csv(test) # for later

    for task in active_tasks:
        print(f"\nFor Task: {task.upper()}\n")

        labels_arr = np.array(all_labels[task])
        preds_arr = np.array(all_preds[task])
        conf_arr = np.array(all_confidences[task])

        print(f"\nClassification Report")
        report = classification_report(
            labels_arr, 
            preds_arr, 
            target_names=label_names[task],
            digits=4,
            zero_division=0
        )
        print(report)

        report_dict = classification_report(
            labels_arr, 
            preds_arr,
            target_names=label_names[task],
            output_dict=True,
            zero_division=0
        )

        correct = (labels_arr == preds_arr)
        mean_conf = conf_arr.mean()
        mean_conf_correct = conf_arr[correct].mean() if correct.any() else 0
        mean_conf_incorrect = conf_arr[~correct].mean() if (~correct).any() else 0

        print(f"Overall Mean confidence: {mean_conf:.4f}")
        print(f"Mean confidence for correct predictions: {mean_conf_correct:.4f}")
        print(f"Incorrect Predictions confidence: {mean_conf_incorrect:.4f}")

        # save summary to JSON
        summary["results"][task] = {
            "macro_f1": float(report_dict["macro avg"]["f1-score"]),
            "macro_precision": float(report_dict["macro avg"]["precision"]),
            "macro_recall": float(report_dict["macro avg"]["recall"]),
            "confidence": {
                "overall": float(mean_conf),
                "correct": float(mean_conf_correct),
                "incorrect": float(mean_conf_incorrect)
            },
            "per_class": report_dict
        }

        # Confusion matrix

        cm = confusion_matrix(labels_arr, preds_arr)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=label_names[task], yticklabels=label_names[task],
            ax=ax
        )
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("True Label", fontweight="bold")
        ax.set_title(f"{task.replace('_', ' ').title()} Confusion Matrix ({args.mode.upper()})", fontweight="bold")
        
        run_name = args.task if args.mode == "stl" else "mtl"
        cm_path = f"outputs/figures/cm_{args.mode}_{args.dataset}_{task}.png"
        fig.savefig(cm_path, dpi = 150, bbox_inches='tight')
        plt.close(fig)
        print("Saved cm to path", cm_path)

        test_df[f'{task}_pred'] = [label_names[task][p] for p in preds_arr] # Map to human readable
        test_df[f'{task}_confidence'] = conf_arr

    # to JSON
    run_name = args.task if args.mode == "stl" else "mtl"
    json_path = f"outputs/eval_summary_{args.mode}_{run_name}_{args.dataset}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved evaluation summary to {json_path}")

    csv_path = f"outputs/test_predictions_{args.mode}_{run_name}_{args.dataset}.csv"
    test_df.to_csv(csv_path, index=False)
    print("Saved raw predictions to CSV at", csv_path)

if __name__ == "__main__":
    main()