# slp_train.py
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from slp import BaseClassifier

# Hyperparameters
in_dim, out_dim = 784, 10
lr = 1e-3  #1e-2, 1e-3, 1e-4
epochs = 60  #20, 40, 60
loss_threshold = 0.1

# Device setup
device = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load dataset
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
classifier = BaseClassifier(in_dim, out_dim).to(device)
optimizer = optim.SGD(classifier.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

def train():
    start_time = time.time()  # Start timer
    classifier.train()
    loss_lt = []

    for epoch in range(epochs):
        running_loss = 0.0
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1).to(device)
            target = target.to(device)

            # Forward pass
            out = classifier(data)
            computed_loss = loss_fn(out, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            computed_loss.backward()
            optimizer.step()

            running_loss += computed_loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_lt.append(avg_loss)
        print(f"Epoch: {epoch + 1}, Train Loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < loss_threshold:
            print(f"Stopping early at epoch {epoch + 1} with loss {avg_loss:.4f}")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Finished in {elapsed_time:.2f} seconds")

    # Plot training loss
    plt.plot(range(1, len(loss_lt) + 1), loss_lt)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"SLP MNIST Training Loss: optimizer SGD, lr {lr}")
    plt.show()

    # Save the model state
    torch.save(classifier.state_dict(), 'mnist_slp.pt')
    print("Saving network in mnist_slp.pt")

if __name__ == "__main__":
    train()
