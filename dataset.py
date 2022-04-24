import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class Resist(Dataset):
    def __init__(self, root='dataset', csv_file='resist2.csv', n_resists=15, normalize=False):
        self.normalize = normalize
        self.csv_path = os.path.join(root, csv_file)
        self.n_resists = n_resists
        
        self.resists, self.depths, self.rhos = self._get_data()
        
    def __len__(self):
        return len(self.resists)

    def __getitem__(self, idx):
        return self.resists[idx], self.depths[idx], self.rhos[idx]
    
    def _get_data(self):
        df = pd.read_csv(self.csv_path, header=None)
        
        n_targets = len(df.columns) - self.n_resists
        n_depths = n_targets//2
        n_rhos = n_targets - n_depths
        
        resists = df.iloc(axis=1)[:self.n_resists].astype('float32').to_numpy()
        depths = df.iloc(axis=1)[self.n_resists:self.n_resists + n_depths].astype('float32').to_numpy()
        rhos = df.iloc(axis=1)[-n_rhos:].astype('float32').to_numpy()
    
        return torch.tensor(resists), torch.tensor(depths), torch.tensor(rhos)