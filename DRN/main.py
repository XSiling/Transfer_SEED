import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import tqdm
# import data_loader
from model import DANN
import os
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
import argparse
import warnings
import json


warnings.filterwarnings("ignore")



parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--test_person', type=int, default=1)
parser.add_argument('--data_path', default='../../')
parser.add_argument('--nepoch', type=int, default=200)
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--result_file', type=str, default='result.txt')
parser.add_argument('--seed', type=int, default=114514)
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(args.seed)


def test(model, target_loader):
    alpha = 0
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_data, t_label) in enumerate(target_loader):
            t_data, t_label = t_data.to(DEVICE), t_label.to(DEVICE)
            t_label = t_label.squeeze()
            class_output, _ = model(input_data=t_data, alpha=alpha)
            prob, pred = torch.max(class_output.data, 1)
            n_correct += (pred == t_label.long()).sum().item()
    acc = float(n_correct) / len(target_loader)
    print("test accuracy:", acc)
    with open(args.result_file, 'a') as f:
        f.write(str(args.test_person))
        f.write('\t')
        f.write(str(acc))
        f.write('\n')


def train(model, optimizer, source_loader, target_loader):
    loss_class = torch.nn.CrossEntropyLoss()
    len_dataloader = min(len(source_loader), len(target_loader))
    for epoch in range(args.nepoch):
        model.train()
        i = 1
        err_s_label_total = 0.0
        err_s_domain_total = 0.0
        err_t_domain_total = 0.0
        err_domain_total = 0.0
        err_total = 0.0
        for (data_src, data_tar) in tqdm(zip(enumerate(source_loader), enumerate(target_loader)), total=len_dataloader, leave=False):
            _, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(
                DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            y_src = y_src.view(-1)
            p = float(i + epoch * len_dataloader) / args.nepoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            class_output, err_s_domain = model.forward(x_src, alpha)
            err_s_domain_total += err_s_domain.item()
            err_s_label = loss_class(class_output, y_src)
            err_s_label_total += err_s_label.item()
            _, err_t_domain = model(
                input_data=x_tar, alpha=alpha, source=False)
            err_t_domain_total += err_t_domain.item()
            err_domain = err_t_domain + err_s_domain
            err_domain_total += err_domain.item()
            err = err_s_label + args.gamma * err_domain
            err_total += err.item()
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            i += 1
            # print(i)
        print("epoch:", epoch, "loss:", err_total)
        torch.save(model.state_dict(), 'model.pth')


def data_loader(test_id):
    assert os.path.exists(args.data_path + 'EEG_X.mat'), 'No Data at' + args.data_path
    assert os.path.exists(args.data_path + 'EEG_Y.mat'), 'No Data at' + args.data_path
    EEG_X = sio.loadmat(args.data_path + 'EEG_X.mat')['X'][0]
    EEG_Y = sio.loadmat(args.data_path + 'EEG_Y.mat')['Y'][0]
    for i in range(EEG_Y.shape[0]):
        for j in range(EEG_Y[i].shape[0]):
            if EEG_Y[i][j] == -1:
                EEG_Y[i][j] = 2

    X_train, y_train, X_test, y_test = None, None, None, None
    for i in range(0, test_id - 1):
        if X_train is None:
            X_train = EEG_X[i]
            y_train = EEG_Y[i]
        else:
            X_train = np.concatenate((X_train, EEG_X[i]))
            y_train = np.concatenate((y_train, EEG_Y[i]))
    for i in range(test_id, 15):
        if X_train is None:
            X_train = EEG_X[i]
            y_train = EEG_Y[i]
        else:
            X_train = np.concatenate((X_train, EEG_X[i]))
            y_train = np.concatenate((y_train, EEG_Y[i]))
    X_test = EEG_X[test_id - 1]
    y_test = EEG_Y[test_id - 1]

    train_scalar, test_scalar = MinMaxScaler(), MinMaxScaler()
    X_train = train_scalar.fit_transform(X_train)
    X_test = test_scalar.fit_transform(X_test)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long().squeeze()
    train_set = TensorDataset(X_train, y_train)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()
    test_set = TensorDataset(X_test, y_test)

    # train_loader, test_loader = DataLoader(train_set), DataLoader(test_set)
    return train_set, test_set

if __name__ == '__main__':
    source_set, target_set = data_loader(args.test_person)
    source_loader = DataLoader(source_set, batch_size=args.batch_size, shuffle=args.shuffle)
    target_loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=args.shuffle)
    test_loader = DataLoader(target_set, batch_size=1, shuffle=False)
    model = DANN(DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(model, optimizer, source_loader, target_loader)
    test(model, test_loader)
