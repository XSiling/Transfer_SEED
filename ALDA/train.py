import argparse
import os
import os.path as osp
import scipy.io as sio

from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import network
import loss
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import lr_schedule
import random
import math

def data_loader(test_id):
    assert os.path.exists(args.data_path + '\EEG_X.mat'), 'No Data at' + args.data_path
    assert os.path.exists(args.data_path + '\EEG_Y.mat'), 'No Data at' + args.data_path
    EEG_X = sio.loadmat(args.data_path + '\EEG_X.mat')['X'][0]
    EEG_Y = sio.loadmat(args.data_path + '\EEG_Y.mat')['Y'][0]
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

def test(loader, model, DEVICE):
    start_test = True
    dataset = loader['test']
    with torch.no_grad():
        iter_test = iter(dataset)
        for i in range(len(dataset)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(DEVICE)
            feature, outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(outputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy

def train(config):
    if config["use_gpu"]:
       DEVICE = torch.device('cuda:{}'.format(config["gpu"])) if torch.cuda.is_available() else torch.device('cpu')
    else:
       DEVICE = torch.device('cpu')
    
    ## prepare data
    dset_loaders = {}
    train_set, test_set = data_loader(config["test_person"])
    dset_loaders["source"] = DataLoader(train_set, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
    dset_loaders["target"] = DataLoader(test_set, batch_size=config["batch_size"], shuffle=config["shuffle"], drop_last=True)
    dset_loaders["test"] = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    class_num = 3

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.to(DEVICE)

    ## add additional network for some methods
    if "ALDA" in args.method:
        ad_net = network.Multi_AdversarialNetwork(base_network.output_num(), 1024, class_num)
    else:
        ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
    ad_net = ad_net.to(DEVICE)
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in range(len(gpus))])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in range(len(gpus))])
        
    loss_params = config["loss"]
    high = loss_params["trade_off"]
    begin_label = False
    writer = SummaryWriter(config["output_path"])

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    loss_value = 0
    loss_adv_value = 0
    loss_correct_value = 0
    for i in tqdm(range(config["num_iterations"]), total=config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"]-1:
            base_network.train(False)
            temp_acc = test(dset_loaders, base_network, DEVICE)
            temp_model = base_network  #nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_step = i
                best_acc = temp_acc
                best_model = temp_model
                checkpoint = {"base_network": best_model.state_dict(), "ad_net": ad_net.state_dict()}
                torch.save(checkpoint, osp.join(config["output_path"], "best_model.pth"))
                print("\n##########     save the best model.    #############\n")
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            writer.add_scalar('precision', temp_acc, i)
            print(log_str)

            print("adv_loss: {:.3f} correct_loss: {:.3f} class_loss: {:.3f}".format(loss_adv_value, loss_correct_value, loss_value))
            loss_value = 0
            loss_adv_value = 0
            loss_correct_value = 0

        if i > config["stop_step"]:
            log_str = "method {}, iter: {:05d}, precision: {:.5f}".format(config["output_path"], best_step, best_acc)
            config["final_log"].write(log_str+"\n")
            config["final_log"].flush()
            break
                 
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.to(DEVICE), inputs_target.to(DEVICE), labels_source.to(DEVICE)
        features_source, outputs_source = base_network(inputs_source)
        if args.source_detach:
            features_source = features_source.detach()
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        loss_params["trade_off"] = network.calc_coeff(i, high=high) #if i > 500 else 0.0
        transfer_loss = 0.0
        
        ad_out = ad_net(features)
        adv_loss, reg_loss, correct_loss = loss.ALDA_loss(DEVICE, ad_out, labels_source, softmax_out,
                    weight_type=config['args'].weight_type, threshold=config['threshold'])
        # whether add the corrected self-training loss
        if "nocorrect" in config['args'].loss_type:
            transfer_loss = adv_loss
        else:
            transfer_loss = config['args'].adv_weight * adv_loss + config['args'].adv_weight * loss_params["trade_off"] * correct_loss
        # reg_loss is only backward to the discriminator
        if "noreg" not in config['args'].loss_type:
            for param in base_network.parameters():
                param.requires_grad = False
            reg_loss.backward(retain_graph=True)
            for param in base_network.parameters():
                param.requires_grad = True
        
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        loss_value += classifier_loss.item() / config["test_interval"]
        loss_adv_value += adv_loss.item() / config["test_interval"]
        loss_correct_value += correct_loss.item() / config["test_interval"]            
        total_loss = classifier_loss + transfer_loss
        total_loss.backward()
        optimizer.step()
        
    checkpoint = {"base_network": temp_model.state_dict(), "ad_net": ad_net.state_dict()}
    torch.save(checkpoint, osp.join(config["output_path"], "final_model.pth"))
    ## test
    base_network.train(False)
    acc = test(dset_loaders, base_network, DEVICE)
    with open('result.txt', 'a') as f:
        f.write(str(args.test_person))
        f.write('\t')
        f.write(str(acc))
        f.write('\n')
    print("accuracy:", acc)
    return best_acc

if __name__ == "__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='ALDA')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='Base', help='the base network(feature extractor)')
    parser.add_argument('--test_person', type=int, default=15, help='the id of person for testing')
    parser.add_argument("--data_path", default=os.path.abspath('.'))
    parser.add_argument('--dset', type=str, default='EEG', help="The dataset or source dataset used")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--trade_off', type=float, default=1.0, help="trade off between supervised loss and self-training loss")
    parser.add_argument('--batch_size', type=int, default=64, help="training batch size")
    parser.add_argument('--threshold', default=0.9, type=float, help="threshold of pseudo labels")
    parser.add_argument('--label_interval', type=int, default=200, help="interval of two continuous pseudo label phase")
    parser.add_argument('--stop_step', type=int, default=0, help="stop steps")
    parser.add_argument('--final_log', type=str, default=None, help="final_log file")
    parser.add_argument('--weight_type', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default='all', help="whether add reg_loss or correct_loss.")
    parser.add_argument('--shuffle', type = bool, default = True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--adv_weight', type=float, default=1.0, help="weight of adversarial loss")
    parser.add_argument('--source_detach', default=False, type=str2bool, help="detach source feature from the adversarial learning")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'

    #set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=True

    # train config
    config = {}
    config['args'] = args
    config['method'] = args.method
    config["use_gpu"] = args.use_gpu
    config["gpu"] = args.gpu_id
    config["test_person"] = args.test_person
    config["num_iterations"] = 100004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["batch_size"] = args.batch_size
    config["shuffle"] = args.shuffle
    if os.path.exists(config["output_path"]):
        print("checkpoint dir exists, which will be removed")
        import shutil
        shutil.rmtree(config["output_path"], ignore_errors=True)
    os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    if len(config['gpu'].split(','))>1:
        args.batch_size = 32*len(config['gpu'].split(','))
        print("gpus:{}, batch size:{}".format(config['gpu'], args.batch_size))

    config["prep"] = {'params':{"resize_size":256, "crop_size":224}}
    config["loss"] = {"trade_off":args.trade_off}
    if "Base" in args.net:
        net = network.BaseNet
        config["network"] = {"name":net, \
            "params":{"use_bottleneck":True, "bottleneck_dim":128, "new_cls":True, } }

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset

    if config["dataset"] == "EEG":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 3
        args.stop_step = 10000
    else:
        raise ValueError('Dataset has not been implemented.')
    if args.lr != 0.001:
        config["optimizer"]["lr_param"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["gamma"] = 0.001
    config["out_file"].write(str(config))
    config["out_file"].flush()
    config["threshold"] = args.threshold
    config["label_interval"] = args.label_interval
    if args.stop_step == 0:
        config["stop_step"] = 10000
    else:
        config["stop_step"] = args.stop_step
    if args.final_log is None:
        config["final_log"] = open('log.txt', "a")
    else:
        config["final_log"] = open(args.final_log, "a")
    train(config)
