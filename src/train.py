# train.py
# some code directly from pytorch docs https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
from datetime import datetime
import torch
import random
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score

from dataset import ReviewDataset
from model import Model

SEED = 4321
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

EPOCHS = 5
PATIENCE = 3

# class weights, training loop and early stopping

# ------------------- Class weights -------------------
# Using weights inversely proportional to class frequencies to avoid majority class bias, 
# prioritize useful bug reports / feature requests
def compute_weights(df, column, device):
    classes = np.unique(df[column])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df[column])
    return torch.tensor(weights, dtype=torch.float).to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Remove randomness
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.manual_seed(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    train = "data/processed/original_train.csv"
    val = "data/processed/original_val.csv"

    train_dataset = ReviewDataset(train, tokenizer)
    val_dataset = ReviewDataset(val, tokenizer)

    training_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = Model().to(device)

    train_df = pd.read_csv(train)
    # move input_ids, attention_mask and labels to device in each batch
    
    # weights
    bug_weights = compute_weights(train_df, 'bug_report', device)
    feature_weights = compute_weights(train_df, 'feature_request', device)
    aspect_weights = compute_weights(train_df, 'aspect', device)    
    aspect_sentiment_weights = compute_weights(train_df, 'aspect_sentiment', device)

    # Move tensors to cpu and conver to numpy for usage with sklearn classification report
    # Use detatch() later for predictions
    print("Bug report class weights:", bug_weights.cpu().numpy())
    print("Feature request class weights:", feature_weights.cpu().numpy())
    print("Aspect class weights:", aspect_weights.cpu().numpy())
    print("Aspect sentiment class weights:", aspect_sentiment_weights.cpu().numpy())
    
    # -------------------- Loss Functions -------------------
    #   for later
    #   1.0 * bug_loss +
    #   1.0 * feature_loss +
    #   0.5 * aspect_loss +
    #   0.5 * sentiment_loss

    criterions = {
        'bug_report': nn.CrossEntropyLoss(weight=bug_weights),
        'feature_request': nn.CrossEntropyLoss(weight=feature_weights),
        'aspect': nn.CrossEntropyLoss(weight=aspect_weights),
        'aspect_sentiment': nn.CrossEntropyLoss(weight=aspect_sentiment_weights)
    }

    # -------------------- Optimizer and scheduler -------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=2e-5,        # change
        weight_decay=0.01 
        )
    
    total_steps = len(training_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps) # 10% of steps for warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )

    # ------------------- Training loop -------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_f1 = 0.0
    patience_counter = 0
    epoch_number = 0



    # Initialize with inf to capture best validation loss easily
    best_vloss = float('inf')

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch_number + 1}")
        model.train(True)

        for step, batch in enumerate(training_loader):
            optimizer.zero_grad()

            # forward pass get logits for each head
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            # compute total loss
            loss = 0
            for task in criterions.keys():
                labels = batch[task].to(device)
                loss += criterions[task](outputs[task], labels)

            total_train_loss = loss.item()

            loss.backward()
            
            # clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step() 
            if step % 50 == 0:
                print(f" Batch {step}/{len(training_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(training_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch_number)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # switch to evaluation mode
        model.eval()

        all_preds = {task: [] for task in criterions.keys()}
        all_labels = {task: [] for task in criterions.keys()}

        with torch.no_grad():
            for batch in validation_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask)
                v_loss = 0.0
                for task in criterions.keys():
                    labels = batch[task].to(device)
                    v_loss += criterions[task](outputs[task], labels).item() # detatch .item(*)
                    preds = torch.argmax(outputs[task], dim=1).cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels.cpu().numpy())
        avg_vloss = v_loss / len(validation_loader)
        writer.add_scalar("Loss/val", avg_vloss, epoch_number)
        print("\nValidation Metrics:")    
        epoch_f1 = []
        for task in criterions.keys():
            task_f1 = f1_score(all_labels[task], all_preds[task], average='macro')
            epoch_f1.append(task_f1)
            writer.add_scalar(f"F1/val_{task}", task_f1, epoch_number)
            print(f" {task} Macro F1: {task_f1:.4f}")
        avg_macro_f1 = np.mean(epoch_f1)
        writer.add_scalar("F1/val_macro_avg", avg_macro_f1, epoch_number)
        print(f" Average Macro F1: {avg_macro_f1:.4f}")

        if avg_macro_f1 > best_f1:
            best_f1 = avg_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"outputs/best_mode.pt")
            print(" New best model saved.")
        else:
            patience_counter += 1
            print(f" No improvement. Patience counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(" Early stopping triggered.")
                break
    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()