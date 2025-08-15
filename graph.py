import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

# Sample training history (accuracy and loss for train and val)
epochs = 5
train_accuracy = [0.9613, 0.9607, 0.9615, 0.9608, 0.9623]
val_accuracy = [0.8846, 0.8870, 0.8851, 0.8876, 0.8859]
train_loss = [0.1576, 0.1565, 0.1553, 0.1540, 0.1533]
val_loss = [0.3448, 0.3418, 0.3430, 0.3433, 0.3419]

# Hyperparameters info
batch_size = 32
train_val_split = "80% train / 20% val"

# Sample confusion matrix data (true labels and predicted labels)
y_true = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]  # 0 = class A, 1 = class B
y_pred = [0, 1, 0, 0, 0, 1, 1, 0, 1, 1]

# Calculate confusion matrix and metrics
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)

# Plot accuracy and loss graphs
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Accuracy plot
axs[0].plot(range(1, epochs+1), train_accuracy, marker='o', label='Training Accuracy')
axs[0].plot(range(1, epochs+1), val_accuracy, marker='o', label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()
axs[0].grid(True)

# Loss plot
axs[1].plot(range(1, epochs+1), train_loss, marker='o', label='Training Loss')
axs[1].plot(range(1, epochs+1), val_loss, marker='o', label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()
axs[1].grid(True)

# Add hyperparameters text to figure
plt.figtext(0.15, 0.92, f'Epochs: {epochs}    Batch size: {batch_size}    Train/Val Split: {train_val_split}', 
            fontsize=12, ha='left')

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()

# Plot confusion matrix with seaborn heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print performance metrics
print(f'Accuracy: {acc:.2f}')
print(f'Precision: {prec:.2f}')
print(f'Recall: {rec:.2f}')
