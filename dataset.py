import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Resist2(Dataset):
    def __init__(self, root='dataset', normalize=False):
        self.normalize = normalize
        self.csv_path = os.path.join(root, 'resist2.csv')
        
        self.resists, self.depths, self.rhos = self._get_data()
        
    def __len__(self):
        return len(self.resists)

    def __getitem__(self, idx):
        return self.resists[idx], self.depths[idx], self.rhos[idx]
    
    def _get_data(self):
        data = pd.read_csv(self.csv_path, header=None).astype(np.float32).to_numpy()
        if self.normalize:
            data = data/data.max(axis=0)
        resists, depths, rhos = data[:, :15], data[:, 15:16], data[:, 16:]
        return torch.tensor(resists), torch.tensor(depths), torch.tensor(rhos)