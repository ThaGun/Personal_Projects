import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Designing a policy network with:
    - Input layer (12, 512)
    - Hidden layer (512, 512)
    - Output layer (512, 2)
'''
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        '''
        Mosst common activation functions Relu
        '''
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Network test
if __name__ == "__main__":
    model = DQN(12, 2)
    state = torch.randn(2, 12)
    output = model(state)
    print(output)