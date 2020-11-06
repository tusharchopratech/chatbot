import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(input_size, hidden_size)
        self.l3 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = nn.functional.relu(self.l1(x))
        out = nn.functional.relu(self.l2(x))
        out = self.l3(x)
        return out
