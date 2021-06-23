import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    
    def __init__(self,):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(2,3)
        
    def forward(self, data):
        print("inside model")
        print(data)

        return self.fc1(data)
