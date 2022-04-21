import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from dataset import Resist2
from models import Encoder, Decoder, AE
from utils import EarlyStopping, train, evaluate, plot_graph


NUM_EPOCHS = 1000
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Resist2(normalize=False)
train_data, val_data = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

encoder = Encoder()
decoder = Decoder()
model = AE(encoder, decoder).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
criterion = nn.MSELoss()

early_stopping = EarlyStopping(verbose=True)

train_losses = []
val_losses = []
for epoch in range(1, NUM_EPOCHS+1):
    train_loss, _, _ = train(model, train_loader, criterion, optimizer, DEVICE)
    train_losses.append(train_loss)
    
    val_loss, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    
    print('-'*20)
    print(f'Epoch: [{epoch:02}/{NUM_EPOCHS}]')
    print(f'Train Loss = {train_loss:.3f}, Val Loss = {val_loss:.3f}')
    
    early_stopping(val_loss, model)
    print('-'*20)
    
    if early_stopping.early_stop:
        print('Early stopping')
        break
    
plot_graph(train_losses, val_losses)