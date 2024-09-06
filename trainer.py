import torch
import properscoring as ps
import numpy as np
from utils import EarlyStopping, aleatoric_loss, mmd_loss
import os
import scipy.stats as st
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, confusion_matrix
import torch.nn.functional as F

def label_smoothing(labels, num_classes, smoothing=0.1):
    confidence = 1.0 - smoothing
    smooth_value = smoothing / (num_classes - 1)
    # Create one-hot labels and apply label smoothing
    one_hot = torch.full((labels.size(0), num_classes), smooth_value, device=labels.device)
    one_hot.scatter_(1, labels.unsqueeze(1), confidence)

    return one_hot


def compute_loss(output, target, num_classes, criterion,  use_label_smoothing=False, smoothing=0.1):
    if use_label_smoothing:
        # Apply label smoothing
        smoothed_labels = label_smoothing(target, num_classes, smoothing=smoothing)

        # Compute log softmax of the model output
        log_probs = F.log_softmax(output, dim=-1)

        # Use KLDivLoss to compute the loss between log_probs and smoothed_labels
        loss = F.kl_div(log_probs, smoothed_labels, reduction='batchmean')
    else:
        loss = criterion(output, target)
    return loss

def origin_train(data, X, Y, model, optim, criterion, args):
    train_batch_losses = []
    model.train()
    for batchX, batchY in data.get_batches(X, Y, args.batch_size):
        output = model(batchX)
        _, predictions = torch.max(output, 1)

        loss = compute_loss(output, batchY,  5, criterion, args.label_smoothing, args.smoothing)
        # loss = criterion(output, batchY)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        train_batch_losses.append(loss.data.item())
        args.wb.log({'train_batch_loss': loss.data.item()})
    return np.average(train_batch_losses)

# sample_number * window * dimension
# 64 * 10 * 15
def origin_eval(data, X, Y, model, criterion, args):
    model.eval()
    predict, test, output_predictions = None, None, None
    n_correct, n_total = 0, 0
    valid_batch_losses = []
    for batchX, batchY in data.get_batches(X, Y, args.batch_size, True):
        output = model(batchX)
        _, predictions = torch.max(output, 1)
        if predict is None:
            predict = output.clone().detach()
            test = batchY
            output_predictions = predictions
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, batchY))
            output_predictions = torch.cat((output_predictions, predictions))

        loss = criterion(output, batchY)
        # loss = compute_loss(output, batchY, 5, criterion, False, args.smoothing)
        valid_batch_losses.append(loss.item())

        temp = (predictions == batchY)
        n_correct += temp.sum().item()
        n_total += batchY.shape[0]

    check_test, check_output = (test.cpu().detach().numpy(),
                                output_predictions.cpu().detach().numpy())
    c_m = confusion_matrix(check_test, check_output)
    accuracy = n_correct / n_total
    result = evaluate_performance(check_test, check_output)
    return np.average(valid_batch_losses), accuracy, result


def evaluate_performance(check_test, check_output):
    mae = mean_absolute_error(check_test, check_output)
    mse = mean_squared_error(check_test, check_output)
    rmse = np.sqrt(mse)

    r2 = r2_score(check_test, check_output)

    # 打印分析结果

    # 返回各项性能指标
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def calibration_train(data, X, Y, model, optim, epoch, args):
    # print('Epoch: %d' % epoch)
    train_batch_losses = []
    mmd_loss_func = mmd_loss()
    model.train()
    for batchX, batchY in data.get_batches(X, Y, args.batch_size):
        output, log_var = model(batchX)
        loss = mmd_loss_func(batchY, output)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        train_batch_losses.append(loss.data.item())

    return np.average(train_batch_losses)

def calibration_val(data, X, Y, model, args):
    model.eval()
    predict, test = None, None
    mmd_loss_func = mmd_loss()
    valid_batch_losses = []
    for batchX, batchY in data.get_batches(X, Y, args.batch_size, False):
        output, log_var = model(batchX)
        if predict is None:
            predict = output.clone().detach()
            test = batchY
        else:
            predict = torch.cat((predict, output.clone().detach()))
            test = torch.cat((test, batchY))
        loss = mmd_loss_func(batchY, output)
        valid_batch_losses.append(loss.item())
    return np.average(valid_batch_losses)

def evaluation(data, X, Y, model, confidence=95):
    pred_mean, logvar = model(X)
    pred_mean, logvar = pred_mean.cpu().data.numpy(), logvar.cpu().data.numpy()
    test_data, test_label = X.cpu().data.numpy(), Y.cpu().data.numpy()
    pre_std = np.sqrt(np.exp(logvar))
    index = data.prediction_window_size

    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), test_label), 1))
    test_label = inversed[:, -test_label.shape[1]:]
    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), pred_mean), 1))
    pred_mean = inversed[:, -pred_mean.shape[1]:]
    inversed = data.scaler.inverse_transform(np.concatenate((test_data.reshape(test_data.shape[0], -1), pre_std), 1))
    pre_std = inversed[:, -pre_std.shape[1]:]

    rmse = np.sqrt(mean_squared_error(pred_mean, test_label))
    conf = float(confidence) / 100
    interval_low, interval_up = st.norm.interval(conf, loc=pred_mean, scale=pre_std)
    interval_low = interval_low.reshape(interval_low.shape[0], -1)
    interval_up = interval_up.reshape(interval_up.shape[0], -1)
    k_l, k_u = interval_low < Y, interval_up > Y
    picp = np.mean(k_l * k_u)
    cali_error = np.around(np.abs(picp-conf), decimals=3)

    return interval_low, interval_up, rmse, cali_error


