import os
import argparse
import time

import os
import warnings
warnings.filterwarnings("ignore")
from utils import Data_utility, makeOptimizer
from tqdm import tqdm
from models import lob, LSTNet, FuturesNet, CNN, GRU, Attention, preTCN
from trainer import origin_train, origin_eval, evaluation, calibration_train, calibration_val
from utils import EarlyStopping, save_model, load_model, RescaledSaliency
from pathlib import Path
import wandb
import json

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--id', type=int, default=300, help='location of the data file')
parser.add_argument('--year', type=int, default=2020, help='location of the data file')
parser.add_argument('--model', type=str, default='preTCN',  help='')
parser.add_argument('--window', type=int, default=16, help='window size')
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--highway_window', type=int, default=3, help='The window size')
parser.add_argument('--kwindow', type=int, default=30, help='The window size')

parser.add_argument('-n_rnns', type=int, default=1, help='depth of the RNN model')
parser.add_argument('-rnn_hid_size', type=int, default=50)
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')
parser.add_argument('-fc_hid_size', type=int, default=24)
parser.add_argument('--dropout', type=float, default=0.3, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--output_fun', type=str, default='tanh')
parser.add_argument('--hidRNN', type=int, default=64, help='number of RNN hidden units each layer')
parser.add_argument('--hidCNN', type=int, default=64, help='number of CNN hidden units (channels)')
parser.add_argument('--CNN_kernel', type=int, default=3, help='the kernel size of the CNN layers')
# SI的curve 还有 sharp率
parser.add_argument('-lr', type=float, default=0.2)
parser.add_argument('-epochs', type=int, default=20)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-lh_scale', type=float, default=1.,
                    help="likelihood scale, shouldn't need tuning")
parser.add_argument('--seed', type=int, default=1111,help='random seed')
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--model_path', type=str, default='./save',
                    help='path to save checkpoints (default: None)')
parser.add_argument('--patience', type=int, default=50, help='model path')
parser.add_argument('--delta', type=int, default=0, help='patience')
parser.add_argument('--verbose', type=bool, default=False, help='print patience information')
parser.add_argument('--grad_clip', type=int, default=10, help='clip gradient')
parser.add_argument('--calib', type=bool, default=True, help='calibration')
parser.add_argument('--weight-decay', type=float, default=0.001, help='clip gradient')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--label_smoothing', type=bool, default=False, help='smoothing')
parser.add_argument('--smoothing', type=float, default=0.08, help='smoothing')

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import torch.nn as nn
    import torch.optim as opt

    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if args.seed:
        torch.cuda.manual_seed(args.seed)

    file_name = "newdata/" + str(args.id) + "_" + str(args.year) + ".npy"
    print(file_name)
    Data = Data_utility(file_name, 0.6, 0.2, device, args)
    wandb.init(project="futures"+str(args.id), mode="disabled", config=args, group=str(args.year), name=args.model)
    args.wb = wandb

    model = eval(args.model)
    model = model.Model(args, Data).to(device)
    print(model.eval())
    nParams = sum([p.nelement() for p in model.parameters()])
    print('number of parameters: %d' % nParams)

    optim = makeOptimizer(model.parameters(), args)
    early_stopping_hnn = EarlyStopping(patience=args.patience,  delta=args.delta, verbose=args.verbose)
    weight = [0.2, 0.2, 0.2, 0.2, 0.2]
    class_weights = torch.FloatTensor(weight).cuda()
    criterion = nn.CrossEntropyLoss(weight = class_weights).to(device)

    scheduler = opt.lr_scheduler.ExponentialLR(optim, gamma=0.99)
    print('Training start')
    best_acc = 0
    best_results = {
        'mae': None,
        'mse': None,
        'rmse': None,
        'r2': None,
        'val_loss': None,
        'epoch': None
    }
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        # scheduler.step()
        train_loss = origin_train(Data, Data.train[0], Data.train[1], model, optim, criterion, args)
        val_loss, val_acc, result = origin_eval(Data, Data.test[0], Data.test[1], model, criterion, args)
        print('epoch {:2d} | time used: {:5.2f}s | train_loss {:4.3f} | '
              'valid loss {:4.3f} | valid acc {:4.3f} | mae {:4.3f} | mse {:4.3f} | rmse {:4.2f} | r2 {:4.2f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_acc, result['mae'], result['mse'], result['rmse'], result['r2']) )
        # args.wb.log({'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc,  'mae': result['mae'], 'mse': result['mse'], 'rmse': result})
        args.wb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'mae': result['mae'],
            'mse': result['mse'],
            'rmse': result['rmse'],
            'r2': result['r2']
        })

        # Save best results
        if val_acc > best_acc:
            best_acc = val_acc
            best_results['mae'] = result['mae']
            best_results['mse'] = result['mse']
            best_results['rmse'] = result['rmse']
            best_results['r2'] = result['r2']
            best_results['val_loss'] = val_loss
            best_results['epoch'] = epoch

        early_stopping_hnn(val_loss, model)
        if early_stopping_hnn.early_stop:
            print("Early stopping for the phase")
            break
        if epoch % 10 == 0:
            save_path = os.path.join(args.model_path, str(args.year) + "_" + str(args.id), args.model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path,  f'checkpoint{epoch}.pt')
            save_model(model, save_file)

    results = {
        'best_acc': best_acc,
        'mae': best_results['mae'],
        'mse': best_results['mse'],
        'rmse': best_results['rmse'],
        'r2': best_results['r2'],
        'val_loss': best_results['val_loss'],
        'epoch': best_results['epoch']
    }
    args.wb.log(results)

    results_file = os.path.join(save_path, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main()
