# visualize.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_metrics(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion(all_labels, all_preds, display_labels):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot(cmap="Blues")
    plt.title('Confusion Matrix')
    plt.show()
