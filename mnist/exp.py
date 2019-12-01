import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import skimage.transform
import mnist

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256, 4)
        self.fc2 = nn.Linear(4, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def downsample(data):
    return torch.from_numpy(np.array([skimage.transform.resize(data[i][0], (16, 16))[None, :] for i in range(len(data))]))

batch_size = 8
test_batch_size = 1000

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


model = Net()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data = downsample(data)
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    print(loss)

