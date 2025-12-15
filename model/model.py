#创建模型
import torch.nn as nn
import torch.nn.functional as F
class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 120)          # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = self.fc2(x)              # output(10)
        return x

