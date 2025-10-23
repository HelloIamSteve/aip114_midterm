import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from config import *
from dataset import Midterm100
from model import *

def train(model, device, train_loader, optimizer, criterion):
    model.train()

    loss_total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()
        
        loss.backward()
        optimizer.step()

    loss_avg = loss_total / len(train_loader)

    return loss_avg

@torch.no_grad()
def valid(model, val_loader, criterion, device):
    model.eval()

    loss_total = 0

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model(inputs)
        loss = criterion(output, labels)
        loss_total += loss.item()

    loss_avg = loss_total / len(val_loader)

    return loss_avg

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using: {device}')

    # dataset
    dataset_dir = os.path.join('Midterm100', 'Midterm100')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    midterm100_train = Midterm100(root_dir=dataset_dir, mode='train',transform=transform)
    midterm100_val = Midterm100(root_dir=dataset_dir, mode='val',transform=transform)

    # data loader
    train_loader = DataLoader(midterm100_train, batch_size=2, shuffle=True)
    val_loader = DataLoader(midterm100_val, batch_size=2)

    # load model
    # model = U_Net().to(device)    #36.71, 0.9783
    model = Improved_U_Net(alpha=0.6, beta=0.4).to(device)  #37.46, 0.9803

    # train
    save_path = f'{model.name}_results'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('make path')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    loss_train_list = []
    loss_val_list = []
    val_loss_best = float('inf')
    suffix = '_MSELoss'
    for epoch_num in tqdm(range(epoch_num), leave=False):
        loss_train = train(model, device, train_loader, optimizer, criterion)
        loss_val = valid(model, val_loader, criterion, device)

        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        if loss_val < val_loss_best:
            model.save_model(f'./{save_path}/{model.name}{suffix}_best.pt', record=False)
            val_loss_best = loss_val

        if not (epoch_num+1) % 100:
            model.save_model(f'./{save_path}/{model.name}{suffix}_{epoch_num+1}.pt', record=False)

    model.save_model(f'./{save_path}/{model.name}{suffix}_last.pt')

    fig = plt.figure()
    plt.title('Training loss')
    plt.plot(loss_val_list, 'r', label='valid')
    plt.plot(loss_train_list, 'b', label='train')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    fig.savefig(f'model_loss{suffix}.png')