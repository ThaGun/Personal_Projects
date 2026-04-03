import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchviz import make_dot
from torchinfo import summary
from model import SystemIDNet
from dataset import load_data
import os
BASE = os.path.dirname(os.path.abspath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, stats = load_data()
model = SystemIDNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

''' NN visualization '''
# x = torch.randn(1, 3)
# y = model(x)
# make_dot(y, params=dict(model.named_parameters())).render("nn_graph", format="png")
summary(model, input_size=(1, 3))

best_val_loss = float("inf")
history = {"train": [], "val": []}

for epoch in range(150):
    '''---Training Phase---'''
    model.train()
    train_loss = 0

    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    "---Validation Phase---"
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            val_loss += criterion(model(X_batch), Y_batch).item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    history["train"].append(train_loss)
    history["val"].append(val_loss)

    ''' Save best checpoints '''
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(BASE, "best_model.pt"))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}  |"
              f"Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f}")

''' Plot '''
e_t = np.arange(0, 150)
fig, axes = plt.subplots(2, 1)
axes[0].plot(e_t, history["train"], label="Train loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_ylim(0, 0.0005)
axes[0].legend()
axes[0].grid()

axes[1].plot(e_t, history["val"], label="Validation loss", color="darkorange")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_ylim(0, 0.0005)
axes[1].legend()
axes[1].grid()

plt.suptitle("Training loss and Validation loss")
plt.tight_layout()
plt.show()
print(f"\nBest val loss: {best_val_loss:.5f}")
torch.save(history, "loss_history.pt")