from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
from scipy.io import loadmat
import scipy
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler
import numpy as np

BATCH_SIZE = 128
scaler = MinMaxScaler()
num_repeat = 1
num_train = 10


class sentimentDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        self.len = data.shape[0]
        
    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return data_tensor, label_tensor
    
    def __len__(self):
        return self.len


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


def load_data(batch_size=BATCH_SIZE, person=1):
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for i in range(1, num_train+1):
        mat_data = loadmat("../train/"+str(i)+".mat")
        train_data_list.append(scaler.fit_transform(mat_data['de_feature']))
        train_label_list.append(mat_data['label'])

    for i in range(11, 14):
        mat_data = loadmat("../test/"+str(i)+".mat")
        test_data_list.append(scaler.fit_transform(mat_data['de_feature']))
        test_label_list.append(mat_data['label'])
        for _ in range(num_repeat-1):
            test_data_list[i-11] = np.concatenate((test_data_list[i-11], scaler.fit_transform(mat_data['de_feature'])))
            test_label_list[i-11] = np.concatenate((test_label_list[i-11], mat_data['label']))

    train_datas = np.concatenate(train_data_list)
    train_labels = np.concatenate(train_label_list)

    trainset = sentimentDataset(train_datas, train_labels)
    dataloader_source = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    testsets = [sentimentDataset(test_data_list[i], test_label_list[i]) for i in range(3)]
    dataloaders_target = [DataLoader(testset, batch_size=batch_size, shuffle=False) for testset in testsets]

    return dataloader_source, dataloaders_target[person-1]


def load_test_data(tar = True, batch_size=BATCH_SIZE, person=1):
    if tar:
        test_data_list = []
        test_label_list = []
        for i in range(11, 14):
            mat_data = loadmat("../test/"+str(i)+".mat")
            test_data_list.append(scaler.fit_transform(mat_data['de_feature']))
            test_label_list.append(mat_data['label'])
        testsets = [sentimentDataset(test_data_list[i], test_label_list[i]) for i in range(3)]
        dataloaders_target = [DataLoader(testset, batch_size=batch_size, shuffle=False) for testset in testsets]
        dataloader = dataloaders_target[person-1]
    else:
        train_data_list = []
        train_label_list = []
        for i in range(1, num_train+1):
            mat_data = loadmat("../train/"+str(i)+".mat")
            train_data_list.append(scaler.fit_transform(mat_data['de_feature']))
            train_label_list.append(mat_data['label'])
        train_datas = np.concatenate(train_data_list)
        train_labels = np.concatenate(train_label_list)
        trainset = sentimentDataset(train_datas, train_labels)
        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    return dataloader