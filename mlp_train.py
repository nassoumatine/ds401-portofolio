# mlp_train.py
import time
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from mlp import BaseClassifier

# Hyperparameters
in_dim, out_dim = 784, 10
feature_dim = 256 #128, 256, 512
loss_threshold = 0.1  # Stop training when loss < 0.1
lr = 1e-3 #1e-2, 1e-3, 1e-4, learning rate
epochs = 40 #20, 40, 60

# Load dataset
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
classifier = BaseClassifier(in_dim, feature_dim, out_dim)
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
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += computed_loss.item()
        loss_lt.append(running_loss/len(train_loader))
        print(f"Epoch: {epoch+1} train loss: {running_loss/len(train_loader)}")
        # avg_loss = running_loss / len(train_loader)
        # print(f"Epoch: {epoch+1} train loss: {avg_loss}")
        # if avg_loss < loss_threshold:  # Early stopping
        #     print(f"Stopping early at epoch {epoch+1} with loss {avg_loss}")
        #     break
    
    end_time = time.time()  
    elapsed_time = end_time - start_time
    print(f"Finished in {elapsed_time:.2f} seconds")


    plt.plot(range(1, epochs+1), loss_lt)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"MNIST Training Loss: optimizer SGD, lr {lr}")
    plt.show()

    # Save the model state
    torch.save(classifier.state_dict(), 'mnist.pt')
    print("Saving network in mnist.pt")

if __name__ == "__main__":
    train()