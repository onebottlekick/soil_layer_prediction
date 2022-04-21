import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_features=15, out_features=3):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(self.fc1.out_features, 64)
        self.fc3 = nn.Linear(self.fc2.out_features, out_features)
        
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, in_features=3, out_features=15):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(self.fc1.out_features, 128)
        self.fc3 = nn.Linear(self.fc2.out_features, out_features)
        
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        return decoded