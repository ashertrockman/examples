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
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def test(model, device, test_loader, train_loader):
    model.eval()

    def get_accuracy(loader):
        correct = 0
        with torch.no_grad():
            for data, target in loader:
                data = downsample(data)
                data, target = data.to(device), target.to(device)
                output = model(data.float())
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return correct / len(loader.dataset)


    nparams = sum(p.numel() for p in model.parameters())
    print("Num params: ", nparams)
    norm = 0
    for p in model.parameters():
        norm += torch.sum(torch.mul(p, p))
    norm = torch.sqrt(norm)
    norm = norm.cpu().float()
    print("Norm: ", norm)
    return nparams, np.asscalar(norm.detach().numpy()), get_accuracy(test_loader), get_accuracy(train_loader)




def downsample(data):
    return torch.from_numpy(np.array([skimage.transform.resize(data[i][0], (16, 16))[None, :] for i in range(len(data))]))

batch_size = 64
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

device = torch.device("cuda")

lams = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0, 5.0, 7.0, 10.0, 50.0, 100.0]
for lam in lams:
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=lam)
    model.train()
    for epoch in range(7):
        print("Epoch %d, lambda = %f" % (epoch, lam))
        for batch_idx, (data, target) in enumerate(train_loader):
            data = downsample(data)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    nparams, l2, test_acc, train_acc = test(model, device, test_loader, train_loader)
    with open("exp-n%d-mnist.csv" % int(nparams), 'a') as f:
        f.write('%f, %f, %f, %f\n' % (l2, test_acc, train_acc, lam))
