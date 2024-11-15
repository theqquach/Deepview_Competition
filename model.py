import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Sample Dataset Class
class VolumetricDataset(Dataset):
    def __init__(self, volumes, labels):
        """
        :param volumes: List of 3D volumetric data (e.g., ultrasound data as numpy arrays)
        :param labels: List of corresponding 3D binary labels (surface vs. non-surface)
        """
        self.volumes = volumes
        self.labels = labels

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        volume = self.volumes[idx]
        label = self.labels[idx]
        return torch.tensor(volume, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.float32)

# 3D CNN Model
class SurfaceNet(nn.Module):
    def __init__(self):
        super(SurfaceNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(128, 1, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool3d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.sigmoid(self.conv4(x))  # Output binary voxel grid
        return x

# Training Loop
def train(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for volume, label in dataloader:
            volume, label = volume.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(volume)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Generate Dummy Data (Replace with Real Data)
num_samples = 10
volumes = [np.random.rand(64, 64, 64) for _ in range(num_samples)]  # Replace with actual volumetric data
labels = [np.random.randint(0, 2, (64, 64, 64)) for _ in range(num_samples)]  # Replace with actual binary labels

# Prepare DataLoader
dataset = VolumetricDataset(volumes, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SurfaceNet().to(device)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
train(model, dataloader, optimizer, criterion, epochs=10)

# Save the Model
torch.save(model.state_dict(), "surface_net.pth")
print("Model saved as 'surface_net.pth'")
