

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_conf_matrix(y_true, y_pred):
    y_test = [label.strip().lower() for label in y_test]
    y_pred = [label.strip().lower() for label in y_pred]
    cm = confusion_matrix(y_true, y_pred, labels=["spam", "ham"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Spam", "Ham"], yticklabels=["Spam", "Ham"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
