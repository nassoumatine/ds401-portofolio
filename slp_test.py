# slp_test.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from slp import BaseClassifier

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load dataset
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model and load saved state
classifier = BaseClassifier(784, 10).to(device)
classifier.load_state_dict(torch.load('mnist_slp.pt'))
classifier.eval()

def test():
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1).to(device)
            target = target.to(device)

            # Forward pass
            out = classifier(data)
            _, preds = out.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Compute accuracy
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test()
