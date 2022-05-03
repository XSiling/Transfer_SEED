import argparse
import torch
from solver import Solver
import os
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
import json

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MCD Implementation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=120, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=2, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--test_person', type=int, default=15, metavar='N',
                    help='the id of person for testing')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--seed', type=int, default=1919810)
parser.add_argument("--data_path", default=os.path.abspath('.'))
parser.add_argument("--result_file", default="result.txt")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


torch.manual_seed(args.seed)


def dataset_load(test_id, batch_size = 64):
    assert os.path.exists(args.data_path + '\EEG_X.mat'), 'No Data at' + args.data_path
    assert os.path.exists(args.data_path + '\EEG_Y.mat'), 'No Data at' + args.data_path
    
    EEG_X = scipy.io.loadmat(args.data_path + '\EEG_X.mat')['X'][0]
    EEG_Y = scipy.io.loadmat(args.data_path + '\EEG_Y.mat')['Y'][0]
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
    
    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) - np.min(X_train, axis=0))
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long().squeeze()
    source_dataset = torch.utils.data.TensorDataset(X_train, y_train)

    X_test = (X_test - np.min(X_test, axis=0)) / (np.max(X_test, axis=0) - np.min(X_test, axis=0))
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long().squeeze()
    target_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return source_dataset, target_dataset

def main():
    solver = Solver(args, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch)

    record_num = args.test_person
    record_train = 'record/k_%s_onestep_%s_%s.txt' % (args.num_k, args.one_step, record_num)
    record_test = 'record/k_%s_onestep_%s_%s_test.txt' % (args.num_k, args.one_step, record_num)
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')

    source_dataset, target_dataset = dataset_load(test_id=args.test_person)

    acc_list = []

    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            if not args.one_step:
                source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                num = solver.train(t, source_loader, target_loader, record_file=record_train)
            else:
                pass
                #num = solver.train_onestep(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
                acc = solver.test(t, target_loader, record_file=record_test, save_model=args.save_model)
                acc_list.append(acc)
                # print(acc)
            if count >= 20000:
                break
    with open(args.result_file, 'a') as f:
        f.write(str(args.test_person))
        f.write('\t')
        f.write(str(acc))
        f.write('\n')
    
if __name__ == '__main__':
    main()