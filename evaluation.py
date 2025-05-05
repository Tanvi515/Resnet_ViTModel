# evaluation.py
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = np.load("data/y_true.npy")
y_pred = np.load("data/y_pred.npy")

acc = accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
prec = precision_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')
rec = recall_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')
f1 = f1_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), average='macro')


print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

mAP = np.mean([average_precision_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])
print(f"Mean Average Precision (mAP): {mAP:.4f}")

conf_matrix = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(10, 8))
for i in range(y_true.shape[1]):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
    class_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {class_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.show()
