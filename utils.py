import os

import torch
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    
    for idx, (resists, depths, rhos) in enumerate(train_loader):
        resists = resists.to(device)
        depths = depths.to(device)
        rhos = rhos.to(device)
        concat_rhos_depths = torch.concat((rhos, depths), dim=1)
        
        optimizer.zero_grad()
        
        latent_vec = model.encoder(resists)        
        latent_loss = criterion(latent_vec, concat_rhos_depths)
        
        ae_outputs = model(resists)
        ae_loss = criterion(ae_outputs, resists)
        
        total_loss = latent_loss + ae_loss
        total_loss.backward()
        
        optimizer.step()
        
    return total_loss.item()/len(train_loader), latent_loss.item()/len(train_loader), ae_loss.item()/len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    
    with torch.no_grad():
        for idx, (resists, depths, rhos) in enumerate(val_loader):
            resists = resists.to(device)
            depths = depths.to(device)
            rhos = rhos.to(device)
            concat_rhos_depths = torch.concat((rhos, depths), dim=1)
            
            latent_vec = model.encoder(resists)        
            latent_loss = criterion(latent_vec, concat_rhos_depths)
        
            ae_outputs = model(resists)
            ae_loss = criterion(ae_outputs, resists)
        
            total_loss = latent_loss + ae_loss
            
    return total_loss.item()/len(val_loader), latent_loss.item()/len(val_loader), ae_loss.item()/len(val_loader)


def plot_graph(train_losses, val_losses):
    plt.plot(train_losses, color='blue', label='train')
    plt.plot(val_losses, color='red', label='validation')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_path='checkpoints'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path

        self.counter = 0
        self.best_score = None
        self.val_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model, save_name='model.pt'):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_model(val_loss, model, save_name)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EearlyStopping counter [{self.counter}/{self.patience}]')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save_model(val_loss, model, save_name)
            self.counter = 0

    def _save_model(self, val_loss, model, save_name):
        os.makedirs(self.save_path, exist_ok=True)
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss:.6f} --> {val_loss:.6f}).')
            print('Saved model.')
        torch.save(model.state_dict(), os.path.join(self.save_path, save_name))
        self.val_loss = val_loss