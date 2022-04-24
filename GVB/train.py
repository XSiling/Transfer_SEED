import argparse
import torch
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import torch.functional as F
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from distutils.version import LooseVersion
from network import BaseNet, AdversarialNetwork
import network
import lr_schedule
import loss






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


def test(loader, model, DEVICE, gvbg=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader["test"])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs.to(DEVICE)
            labels.to(DEVICE)
            _, outputs, _ = model(inputs, gvbg=gvbg)

            if start_test:
                all_output = outputs.float()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float()),0)
                all_label = torch.cat((all_label, labels.float()),0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        return accuracy


def train(config):
    global iter_source, iter_target
    if config["use_gpu"]:
       DEVICE = torch.device('cuda:{}'.format(config["gpu_id"])) if torch.cuda.is_available() else torch.device('cpu')
    else:
       DEVICE = torch.device('cpu')

    #set datasets
    dset_loaders = {}
    train_set, test_set = data_loader(config["test_person"])
    dset_loaders["source"] = DataLoader(train_set, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
    dset_loaders["target"] = DataLoader(test_set, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
    dset_loaders["test"] = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    #set network
    class_num = config["network"]["params"]["class_num"]
    base_network = BaseNet()
    base_network.to(DEVICE)

    #adversarialNetwork
    ad_net = AdversarialNetwork(class_num, 1024)
    ad_net.to(DEVICE)

    #set optimizer
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **optimizer_config["optim_params"])
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_params"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    #train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        #ignore test or save model
        base_network.train(True)
        ad_net.train(True)
        loss_params = config["loss"]
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        #network
        inputs_source, labels_source = iter_source.next()
        inputs_target, _ = iter_target.next()
        inputs_source.to(DEVICE)
        labels_source.to(DEVICE)
        inputs_target.to(DEVICE)
        features_source, outputs_source, focal_source = base_network(inputs_source, gvbg=config["GVBG"])
        features_target, outputs_target, focal_target = base_network(inputs_target, gvbg=config["GVBG"])
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        focals = torch.cat((focal_source, focal_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)

        #loss calculation
        transfer_loss, mean_entropy, gvbg, gvbd = loss.GVB([softmax_out, focals], ad_net, network.calc_coeff(i), GVBD=config["GVBD"])
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + config["GVBG"] * gvbg + abs(config["GVBD"]) * gvbd

        if i % config["print_num"] == 0:
            log_str = "iter: {:05d}, transferloss: {:.5f}, classifier_loss: {:.5f}, mean entropy:{:.5f}, gvbg:{:.5f}, gvbd:{:.5f}".format(i, transfer_loss, classifier_loss, mean_entropy, gvbg, gvbd)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
        print(">>training epoch:", str(i), "loss:", total_loss.item())
        total_loss.backward()
        optimizer.step()

    #test
    base_network.train(False)
    acc = test(dset_loaders, base_network, DEVICE, gvbg=config["GVBG"])
    with open('output.txt', 'a') as f:
        f.write(str(args.test_person))
        f.write('\t')
        f.write(str(acc))
        f.write('\n')
    print("accuracy:", acc)


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='device id to run')
    parser.add_argument('--print_num', type=int, default=10, help='interval of two print loss')
    parser.add_argument('--num_iterations', type=int, default=10000, help='iteration_num')
    parser.add_argument('--output_dir', type=str, default='san', help='output directory of our model')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--trade_off', type=float, default=1, help='parameter for transfer loss')
    parser.add_argument('--GVBG', type=float, default=1, help='lambda: parameter for GVBG (if lambda==0 then GVBG is not utilized)')
    parser.add_argument('--GVBD', type=float, default=0, help='mu: parameter for GVBD (if mu==0 then GVBD is not utilized)')
    parser.add_argument('--CDAN', type=bool, default=False, help='utilize CDAN or not')
    parser.add_argument("--test_person", type=int, default=15)
    parser.add_argument("--use_gpu", default=True)
    parser.add_argument("--test_interval", type=int, default=100)
    parser.add_argument("--snapshot_interval", type=int, default=5000)
    parser.add_argument("--data_path", type=str, default="../../")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--seed", type=int, default=114514)
    args = parser.parse_args()

    #train_config
    config = {}
    config["GVBG"] = args.GVBG
    config["GVBD"] = args.GVBD
    config["CDAN"] = args.CDAN
    config["use_gpu"] = args.use_gpu
    config["gpu"] = args.gpu_id
    config["test_person"] = args.test_person
    config["num_iterations"] = args.num_iterations
    config["print_num"] = args.print_num
    config["shuffle"] = args.shuffle
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["batch_size"] = args.batch_size

    if not os.path.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    config["out_file"] = open(os.path.join(config['output_path'], "log.txt"), "w")
    if not os.path.exists(config['output_path']):
        os.mkdir(config["output_path"])

    config["prep"] = {"params": {"resize_size": 256, "crop_size": 224, "alexnet": False}}
    config["loss"] = {"trade_off": args.trade_off}

    config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, 'momentum': 0.9, \
                            'weight_decay':0.0005, 'nesterov': True}, 'lr_type': 'inv', \
                           'lr_params': {'lr': args.lr, 'gamma': 0.001, 'power': 0.75}}
    config["network"] = {"params": {"class_num": 3}}
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #torch.backends.cudnn.determinstic = True
    #random.seed(seed)

    config["out_file"].write(str(config))
    config["out_file"].flush()
    train(config)