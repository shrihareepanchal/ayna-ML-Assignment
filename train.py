import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import UNet
from utils import PolygonDataset
import wandb

wandb.init(project="ayna-polygon-coloring")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = PolygonDataset("dataset/training/data.json", "dataset/training/inputs", "dataset/training/outputs")
val_data = PolygonDataset("dataset/validation/data.json", "dataset/validation/inputs", "dataset/validation/outputs")

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

model = UNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for img, out, color in train_loader:
        img, out, color = img.to(device), out.to(device), color.to(device)
        optimizer.zero_grad()
        pred = model(img, color)
        loss = criterion(pred, out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    wandb.log({"epoch": epoch+1, "loss": running_loss/len(train_loader)})

torch.save(model.state_dict(), "unet_model.pth")
