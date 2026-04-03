import torch
import torch.nn as nn

class SystemIDNet(nn.Module):
    def __init__(self, in_dim=3, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)

''' Qiuck sanity check '''
if __name__ == "__main__":
    model = SystemIDNet()
    dummy = torch.randn(8, 3)   # 8 samples
    out = model(dummy)
    print(f"Input shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")