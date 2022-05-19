#experiments with CIFAR-10 

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import tt_linearV2

import time
import os
import copy
from prettytable import PrettyTable
import torch.onnx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class TT_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = tt_linearV2.tt_LinearV2(np.array([4, 8, 4, 8, 3]), np.array([8, 8, 8, 8, 8]), np.array([1, 4, 4, 4, 4, 1]))
        self.fc2 = tt_linearV2.tt_LinearV2(np.array([8, 8, 8, 8, 8]), np.array([8, 4, 8, 4, 8]), np.array([1, 4, 4, 4, 4, 1]))
        self.fc3 = tt_linearV2.tt_LinearV2(np.array([8, 4, 8, 4, 8]), np.array([4, 4, 4, 4, 4]), np.array([1, 4, 4, 4, 4, 1]))
        self.fc4 = tt_linearV2.tt_LinearV2(np.array([4, 4, 4, 4, 4]), np.array([4, 4, 4, 4, 4]), np.array([1, 4, 4, 4, 4, 1]))
        #self.fc5 = tt_linearV2.tt_LinearV2(np.array([4, 4, 4, 4, 4]), np.array([5, 2, 1, 1, 1]), np.array([1, 4, 4, 4, 4, 1]))
        self.fc5 = nn.Linear(1024, 10)
        self.bn1 = nn.BatchNorm1d(32768)
        self.bn2 = nn.BatchNorm1d(8192)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(10)
        self.drop = nn.Dropout(0.15)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        x = self.drop(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.bn5(self.fc5(x)))
        return F.log_softmax(x, dim=1)

class Usual_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 32768)
        self.fc2 = nn.Linear(32768, 8192)
        self.fc3 = nn.Linear(8192, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 10)
        self.bn1 = nn.BatchNorm1d(32768)
        self.bn2 = nn.BatchNorm1d(8192)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(10)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.fc1(x))))
        x = self.drop(F.relu(self.bn2(self.fc2(x))))
        x = self.drop(F.relu(self.bn3(self.fc3(x))))
        x = self.drop(F.relu(self.bn4(self.fc4(x))))
        x = F.relu(self.bn5(self.fc5(x)))
        return F.log_softmax(x, dim=1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    lss_train = []
    lss_test = []
    acc_test = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            cur_set = trainset if phase == 'train' else testset 
            for inputs, labels in cur_set:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.reshape(-1, 32*32*3))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler != None:  
                scheduler.step()

            epoch_loss = running_loss / len(cur_set)
            epoch_acc = running_corrects.double() / len(cur_set)

            cur_loss = lss_train if phase == 'train' else lss_test
            cur_loss += [epoch_loss]
            if phase == 'test':
                acc_test += [epoch_acc.item()]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #print('train: {}'.format(lss_train))
    #print('test: {}'.format(lss_test))
    #print('acc: {}'.format(acc_test))

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    train = datasets.CIFAR10('./datasets', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ]))
    test = datasets.CIFAR10('./datasets', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))

    transf = transforms.Compose([
        ])

    trainset = torch.utils.data.DataLoader(train, batch_size=100, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=100, shuffle=False)

    tt_model = TT_Net()
    count_parameters(tt_model)
    tt_model.to(device)
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(tt_model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.RMSprop(tt_model.parameters(), lr=0.01, momentum=0.0)
    optimizer = optim.Adadelta(tt_model.parameters(), lr=2.0)
    #optimizer = optim.Adam(tt_model.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)
    tt_model = train_model(tt_model, criterion, optimizer, None, num_epochs=100)
    torch.save(tt_model, 'saved_model.pth')