from time import time
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df = pd.read_csv('data/labeled_comments.csv')  # Ensure this CSV has 'Comment' and 'Label' columns

df['Comment'] = df['Comment'].fillna("")
df['Label'] = df['Label'].map({'Not Cyberbullying': 0, 'Cyberbullying': 1})

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Comment'].astype(str), df['Label'], test_size=0.2, random_state=42
)

# Load BERT tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_texts(texts):
    return tokenizer(list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt")

# Tokenize training and validation data
train_encodings = tokenize_texts(train_texts)
val_encodings = tokenize_texts(val_texts)

# Custom Dataset class
class CommentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Create PyTorch Datasets
train_dataset = CommentDataset(train_encodings, train_labels.values)
val_dataset = CommentDataset(val_encodings, val_labels.values)

# Load BERT model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Define evaluation metrics
def compute_metrics(pred):
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids
    
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Detailed report and confusion matrix
    report = classification_report(labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'confusion_matrix': conf_matrix
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
start_time = time()

# Train model
train_result = trainer.train()
train_duration = time() - start_time
trainer.save_model("./bert")

# Log training loss
train_loss = train_result.training_loss

# Evaluate model
start_eval = time()
eval_result = trainer.evaluate()
eval_duration = time() - start_eval

# Extract evaluation metrics directly from eval_result
accuracy = eval_result['eval_accuracy']
precision = eval_result.get('eval_precision', 'N/A')
recall = eval_result.get('eval_recall', 'N/A')
f1 = eval_result.get('eval_f1', 'N/A')
eval_loss = eval_result['eval_loss']
    
# Extract classification report and confusion matrix
classification_rep = eval_result.get('eval_classification_report', 'N/A')
confusion_mat = eval_result.get('eval_confusion_matrix', 'N/A')

trainer.save_model("./bert")

# Save metrics to a file
with open("metrics_output.txt", "w") as f:
    f.write(f"Training Duration: {train_duration} seconds\n")
    f.write(f"Evaluation Duration: {eval_duration} seconds\n")
    f.write(f"Training Loss: {train_loss}\n")
    f.write(f"Evaluation Loss: {eval_loss}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write("Classification Report:\n")
    f.write(f"{classification_rep}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{confusion_mat}\n")

# Print metrics to console as well
print(f"Training Duration: {train_duration} seconds")
print(f"Evaluation Duration: {eval_duration} seconds")
print(f"Training Loss: {train_loss}")
print(f"Evaluation Loss: {eval_loss}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(confusion_mat)
print(eval_result)


# Generate labels for the entire dataset if needed
def label_comments(texts):
    encodings = tokenize_texts(texts)
    dataset = CommentDataset(encodings, [0]*len(texts))
    loader = DataLoader(dataset, batch_size=8)

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

    return predictions

# If needed, label the entire dataset (already labeled)
df['Predicted_Label'] = label_comments(df['Comment'])
df['Predicted_Label'] = df['Predicted_Label'].map({0: "Not Cyberbullying", 1: "Cyberbullying"})

# Display the first few rows of the dataset
df.to_csv("data/bert_prediction.csv", index=False)
print(df.head())


