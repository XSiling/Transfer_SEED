import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import tqdm
import data_loader
from sklearn.preprocessing import MinMaxScaler
from model import DANN
import os
import scipy.io as sio
from torch.utils.data import DataLoader, TensorDataset
import argparse
import warnings
warnings.filterwarnings("ignore")
import json


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=.5)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument("--data_path", default='../../')
parser.add_argument('--test_person', type=int, default=15)
parser.add_argument('--nepoch', type=int, default=200) #默认是100
parser.add_argument('--model_path', type=str, default='models')
parser.add_argument('--result_path', type=str, default='result/result.csv')
parser.add_argument('--seed', type=int, default=1919810)
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(args.seed)


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


def test(model, tar = True):
    alpha = 0
    dataloader = data_loader.load_test_data(tar=tar, person=args.person)
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_data, t_label) in enumerate(dataloader):
            t_data, t_label = t_data.to(DEVICE), t_label.to(DEVICE)
            t_label = t_label.squeeze()
            class_output, _ = model(input_data=t_data, alpha=alpha)
            prob, pred = torch.max(class_output.data, 1)
            n_correct += (pred == t_label.long()).sum().item()
    acc = float(n_correct) / len(dataloader.dataset) * 100
    return acc


def train(model, optimizer, dataloader_src, dataloader_tar):
    loss_class = torch.nn.CrossEntropyLoss()
    best_acc = -float('inf')
    len_dataloader = min(len(dataloader_src), len(dataloader_tar))
    train_acc_list = []
    test_acc_list = []
    for epoch in range(args.nepoch):
        model.train()
        i = 1
        err_s_label_total = 0.0
        err_s_domain_total = 0.0
        err_t_domain_total = 0.0
        err_domain_total = 0.0
        err_total = 0.0
        source_set, target_set = data_loader(args.test_person)
        dataloader_src = DataLoader(source_set, batch_size=args.batch_size, shuffle=True)
        dataloader_tar = DataLoader(target_set, batch_size=args.batch_size, shuffle=True)
        for (data_src, data_tar) in tqdm.tqdm(zip(enumerate(dataloader_src), enumerate(dataloader_tar)), total=len_dataloader, leave=False):
            _, (x_src, y_src) = data_src
            _, (x_tar, _) = data_tar
            x_src, y_src, x_tar = x_src.to(
                DEVICE), y_src.to(DEVICE), x_tar.to(DEVICE)
            y_src = y_src.view(-1)
            p = float(i + epoch * len_dataloader) / args.nepoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            class_output, err_s_domain = model(input_data=x_src, alpha=alpha)
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
        item_pr = 'Epoch: [{}/{}], classify_loss: {:.4f}, domain_loss_s: {:.4f}, domain_loss_t: {:.4f}, domain_loss: {:.4f},total_loss: {:.4f}'.format(
            epoch, args.nepoch, err_s_label_total, err_s_domain_total, err_t_domain_total, err_domain_total, err_total)
        print(item_pr)
        fp = open(args.result_path, 'a')
        fp.write(item_pr + '\n')

        # test
        acc_src = test(model, tar = False)/100.
        acc_tar = test(model, tar = True)/100.
        train_acc_list.append(acc_src)
        test_acc_list.append(acc_tar)
        test_info = 'Source acc: {:.4f}, target acc: {:.4f}'.format(acc_src, acc_tar)
        fp.write(test_info + '\n')
        print(test_info)
        fp.close()
        
        if best_acc < acc_tar:
            best_acc = acc_tar
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            torch.save(model, '{}/save_model.pth'.format(args.model_path))
    print('Test acc: {:.4f}'.format(best_acc))
    return train_acc_list, test_acc_list


if __name__ == '__main__':
    loader_src, loader_tar = data_loader(args.test_person)
    # loader_src, loader_tar = data_loader.load_data(person=args.person)
    model = DANN(DEVICE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_acc_list, test_acc_list = train(model, optimizer, loader_src, loader_tar)
    jd = {"train_acc": train_acc_list, "test_acc": test_acc_list}
    with open(str(args.seed)+'/acc'+str(args.person)+'.json', 'w') as f:
        json.dump(jd, f)
