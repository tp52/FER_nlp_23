import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import pandas as pd
from sklearn.metrics import  hamming_loss
import numpy as np
import os
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score


os.environ["WANDB_DISABLED"] = "true"

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions", num_labels=12,ignore_mismatched_sizes=True)  # Set the correct number of labels

# Load your dataset
data = pd.read_csv("../data/final-combined-ds.csv")

# Feature and target variables
X = list(data["lyric"])
emotion_columns = ['anger', 'confidence', 'desire', 'disgust', 'gratitude',
                   'joy', 'love', 'lust',  'sadness', 'shame',
                   'fear',  'anticipation']
y = data[emotion_columns].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

# Define the custom dataset
class Song_Dataset(Dataset):
    def __init__(self, lyrics, labels, tokenizer, max_len=512):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyrics = str(self.lyrics[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        encoding = self.tokenizer(lyrics, truncation=True, padding="max_length", max_length=self.max_len, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }


train_dataset = Song_Dataset(X_train, y_train, tokenizer)
val_dataset = Song_Dataset(X_val, y_val, tokenizer)

# Define Trainer parameters
def multi_labels_metrics(predictions, labels, column_names, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))

    # Calculate the predicted labels based on a threshold
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1

    hamming = hamming_loss(labels, y_pred)

    metrics = {
        "hamming_loss": hamming,
    }

    # Calculate precision, recall, and F1 score for each emotion
    for i, column_name in enumerate(column_names):
        precision = precision_score(labels[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(labels[:, i], y_pred[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], y_pred[:, i], zero_division=0)

        metrics[f"{column_name}_precision"] = precision
        metrics[f"{column_name}_recall"] = recall
        metrics[f"{column_name}_f1"] = f1

    return metrics



def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_labels_metrics(predictions=preds, labels=p.label_ids, column_names=emotion_columns)
    return result


args = TrainingArguments(
    per_device_train_batch_size=6,
    per_device_eval_batch_size=2,
    output_dir='/results',
    num_train_epochs=7,
    save_steps=65,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()
print(results)