# train.py
# Training script for both MTL and STL setups 
# Structure adapted and adjusted from standard PyTorch training loops
import argparse
import os
from datetime import datetime
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import ReviewDataset
from model import Model, SingleTaskModel



# Fixed seed for near reproducibile runs
SEED = 4321
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



def compute_weights(df, column, device):
    """Computes inverse frequency class weights for a label column"""
    classes = np.unique(df[column])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df[column])
    return torch.tensor(weights, dtype=torch.float).to(device)


def parse_args():
    parser = argparse.ArgumentParser(description="RECLASS, Multitask learning for review classification.")
    parser.add_argument("--mode", type=str, default="mtl", choices=["mtl", "stl"], help="Choose between 'mtl' (multitask learning) and 'stl' (single task learning).")
    parser.add_argument("--task", type=str, default="all", choices=["all", "bug_report", "feature_request", "aspect", "aspect_sentiment"], help="Specific task to train for stl usage only" )
    parser.add_argument("--dataset", type=str, default="original", choices=["original", "boosted"], help="Choose between 'original' and 'boosted' dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Keep to 16 or 8 for 8GB VRAM")
    parser.add_argument("--epochs", type=int, default=5, help="Maxiumum training epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting training...", flush=True)
    print("Using device:", device)

    # Set cuda seeds for reproducibility on GPU
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.manual_seed(SEED)
    print(f"Using dataset: {args.dataset.upper()}")

    # Force deterministic for reproducibility at a slight performance cost
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data into train/val splits
    train = f"data/processed/{args.dataset}_train.csv"
    val = f"data/processed/{args.dataset}_val.csv"
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # Tokenizer initilization
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    train_dataset = ReviewDataset(train, tokenizer)
    val_dataset = ReviewDataset(val, tokenizer)

    training_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Shared model uses encoder across all tasks, STL model trains one task at a time
    if args.mode == "mtl":
        model = Model().to(device)
        active_tasks = ['bug_report', 'feature_request', 'aspect', 'aspect_sentiment']
        run_name = f"mtl_{args.dataset}"
    else:
        if args.task == "all":
            raise ValueError("For single task learning, please specify a task using --task argument.")
        
        task_classes = {
            'bug_report': 2,
            'feature_request': 2,
            'aspect': 6,
            'aspect_sentiment': 3
            }
        model = SingleTaskModel(args.task, task_classes[args.task]).to(device)
        active_tasks = [args.task]
        run_name = f"stl_{args.task}_{args.dataset}"
    
    train_df = pd.read_csv(train)
    
    # Compute per-task weights from the training split
    print("\n Computing class weights...")
    bug_weights = compute_weights(train_df, 'bug_report', device)
    feature_weights = compute_weights(train_df, 'feature_request', device)
    aspect_weights = compute_weights(train_df, 'aspect', device)    
    aspect_sentiment_weights = compute_weights(train_df, 'aspect_sentiment', device)
    
    print("Bug report class weights:", bug_weights.cpu().numpy())
    print("Feature request class weights:", feature_weights.cpu().numpy())
    print("Aspect class weights:", aspect_weights.cpu().numpy())
    print("Aspect sentiment class weights:", aspect_sentiment_weights.cpu().numpy())
    
    # equal weighted task losses. unequal was considered but equal weights performed well without adding complexity
    # CrossEntropyLoss = LogSoftmax + NLLLoss (negative log likelihood) 
    criterions = {
        'bug_report': nn.CrossEntropyLoss(weight=bug_weights),
        'feature_request': nn.CrossEntropyLoss(weight=feature_weights),
        'aspect': nn.CrossEntropyLoss(weight=aspect_weights),
        'aspect_sentiment': nn.CrossEntropyLoss(weight=aspect_sentiment_weights)
    }

    # -------------------- Optimizer and scheduler -------------------
    # adaptive momentum and weight decay keeps track of previous weight adaptions and ensures they dont get too large (weight also shrinks towards 0 each pass) 
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,        
        weight_decay=0.01 
        )
    
    total_steps = len(training_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps) # 10% of steps for warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # Entry point for training loop, with Tensorboard logging and early stopping based on validation macro F1 score
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/reclass_{run_name}_{timestamp}')

    best_f1 = 0.0
    patience_counter = 0

    # Initialize with inf to capture best validation loss easily
    best_vloss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        model.train(True)

        total_train_loss = 0.0

        for step, batch in enumerate(training_loader):
            optimizer.zero_grad()

            # forward pass get logits for each head
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Multitask forward pass
            outputs = model(input_ids, attention_mask)
            
            loss = 0
            for task in active_tasks:
                labels = batch[task].to(device)
                loss += criterions[task](outputs[task], labels)

            total_train_loss += loss.item()

            loss.backward()
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step() 

            if step % 50 == 0:
                print(f" Batch {step}/{len(training_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(training_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch) 
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0.0

        all_preds = {task: [] for task in active_tasks}
        all_labels ={task: [] for task in active_tasks}

        with torch.no_grad():
            for batch in validation_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask)

                v_loss = 0.0 # batch validation loss
                for task in active_tasks:
                    labels = batch[task].to(device)
                    v_loss += criterions[task](outputs[task], labels).item() # detatch .item(*)

                    preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels.cpu().numpy())
                total_val_loss += v_loss     

        avg_vloss = total_val_loss / len(validation_loader)
        writer.add_scalar("Loss/val", avg_vloss, epoch)

        # Performance evaluation summary
        print("\nValidation Metrics (MACRO F1):")    
        epoch_f1 = []
        for task in active_tasks:
            task_f1 = f1_score(all_labels[task], all_preds[task], average='macro', zero_division=0)
            epoch_f1.append(task_f1)
            writer.add_scalar(f"F1/val_{task}", task_f1, epoch)
            print(f" {task}: {task_f1:.4f}")

        avg_macro_f1 = np.mean(epoch_f1)
        writer.add_scalar("F1/val_macro_avg", avg_macro_f1, epoch)
        print(f" Average Macro F1: {avg_macro_f1:.4f}")

        # Early stopping
        if avg_macro_f1 > best_f1:
            best_f1 = avg_macro_f1
            patience_counter = 0
            # Save the model with a name for the type of dataset and epoch for later analysis
            model_save_path = f"outputs/best_model_{run_name}.pt"
            torch.save(model.state_dict(), model_save_path)
            print(" New best model saved to:", model_save_path)
        else:
            patience_counter += 1
            print(f" No improvement. Patience counter: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(" Early stopping triggered.")
                break

    writer.close()
    print("Training complete.")
    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    if torch.cuda.is_available():
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / (1024**3)} GB")

if __name__ == "__main__":
    main()