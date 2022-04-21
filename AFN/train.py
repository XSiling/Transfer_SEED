import os
import numpy as np
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from net import BackboneNetwork, Classifier

# import data_loader

parser = argparse.ArgumentParser()
parser.add_argument("--test_person", type=int, default=15)
parser.add_argument("--save_result", type=str, default=True)
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--pre_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--use_gpu", default=False)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--shuffle", default=True)
parser.add_argument("--weight_L2norm", default=0.05)
parser.add_argument("--lr", default=1e-3)
parser.add_argument("--data_path", default='../../')
args = parser.parse_args()

if args.use_gpu:
    DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
else:
    DEVICE = torch.device('cpu')

netG = BackboneNetwork().to(DEVICE)
netF = Classifier().to(DEVICE)

optimG = optim.SGD(netG.parameters(), lr=args.lr, weight_decay=0.0005)
optimF = optim.SGD(netF.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)


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


def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(F.log_softmax(pred), gt)
    return cls_loss


def get_L2norm_loss_self_driven(x):
    l = (x.norm(p=2, dim=1).mean() - 9) ** 2
    return args.weight_L2norm * l


source_set, target_set = data_loader(args.test_person)
source_loader = DataLoader(source_set, batch_size=args.batch_size, shuffle=args.shuffle)
target_loader = DataLoader(target_set, batch_size=args.batch_size, shuffle=args.shuffle)


def train():
    for epoch in range(1, args.pre_epochs + 1):
        for i, (s_data, s_label) in tqdm(enumerate(source_loader)):
            if s_data.size(0) != args.batch_size:
                continue

            s_data = Variable(s_data.to(DEVICE))
            s_label = Variable(s_label.to(DEVICE))

            optimG.zero_grad()
            optimF.zero_grad()

            s_bottleneck = netG(s_data)
            s_fc2_emb, s_logit = netF(s_bottleneck)

            s_fc2_ring_loss = get_L2norm_loss_self_driven(s_fc2_emb)
            s_cls_loss = get_cls_loss(s_logit, s_label)

            loss = s_cls_loss + s_fc2_ring_loss
            loss.backward()

            optimG.step()
            optimF.step()

    for epoch in range(1, args.epoch + 1):
        source_loader_iter = iter(source_loader)
        target_loader_iter = iter(target_loader)
        print(">>training epoch:" + str(epoch))

        for i, (t_data, _) in tqdm(enumerate(target_loader_iter)):
            try:
                s_data, s_label = source_loader_iter.next()
            except:
                source_loader_iter = iter(source_loader)
                s_data, s_label = source_loader_iter.next()

            if s_data.size(0) != args.batch_size or t_data.size(0) != args.batch_size:
                continue

            s_data = Variable(s_data.to(DEVICE))
            s_label = Variable(s_label.to(DEVICE))
            t_data = Variable(t_data.to(DEVICE))

            optimG.zero_grad()
            optimF.zero_grad()

            s_bottleneck = netG(s_data)
            t_bottleneck = netG(t_data)
            s_fc2_emb, s_logit = netF(s_bottleneck)
            t_fc2_emb, t_logit = netF(t_bottleneck)

            s_cls_loss = get_cls_loss(s_logit, s_label)
            s_fc2_L2norm_loss = get_L2norm_loss_self_driven(s_fc2_emb)
            t_fc2_L2norm_loss = get_L2norm_loss_self_driven(t_fc2_emb)

            loss = s_cls_loss + s_fc2_L2norm_loss + t_fc2_L2norm_loss
            loss.backward()

            optimG.step()
            optimF.step()

        if epoch % 10 == 0:
            torch.save(netG.state_dict(), "./model/netG"+str(epoch)+".pth")
            torch.save(netF.state_dict(), "./model/netF"+str(epoch)+".pth")

train()
