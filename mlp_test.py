# mlp_test.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlp import BaseClassifier

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classifier = BaseClassifier(784, 256, 10)  
classifier.load_state_dict(torch.load('mnist.pt'))

in_dim = classifier.classifier[0].in_features
feature_dim = classifier.classifier[0].out_features
out_dim = classifier.classifier[2].out_features

# Reinitialize model with correct dimensions
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
classifier.load_state_dict(torch.load('mnist.pt'))

def test():
    classifier.eval()
    # accuracy = 0.0
    # computed_loss = 0.0
    # loss_fn = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)
            # computed_loss += loss_fn(out, target)
            # accuracy += torch.sum(preds == target)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

        
        # print(f"Test loss: {computed_loss.item()/(len(test_loader)*64)}, test accuracy: {accuracy*100.0/(len(test_loader)*64)}")
        # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_targets)).float().mean()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    test()