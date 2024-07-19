import torch.nn  as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):
    """ConvNet -> Max_Pool -> RELU -> ConvNet -> 
    Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self, DATASET):
        super(CNN, self).__init__()
        self.DATASET=DATASET
        if DATASET == "mnist":
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(800, 500)
            self.fc2 = nn.Linear(500, 10)
            # initialize weights
            nn.init.xavier_uniform_(self.conv1.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.conv2.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc1.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('relu'))
        else:
            self.conv1 = nn.Conv2d(3, 64, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(64, 64, 5)
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(64 * 5 * 5, 120)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(120, 64)
            )
            self.fc3 = nn.Linear(64, 10)
            nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, x):
        if self.DATASET == "mnist":
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, int(np.prod(x.shape[1:])))
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.DATASET == "cifar101":
            x = self.resnet18(x)
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return x