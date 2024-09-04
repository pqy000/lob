import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m - args.prediction_window_size
        self.hw = args.highway_window
        self.prediction_window_size = args.prediction_window_size
        self.activate1=F.relu
        self.hidR = args.rnn_hid_size
        self.fc_hid_size = args.fc_hid_size
        self.rnn1 = nn.GRU(self.variables,self.hidR,num_layers=args.rnn_layers)
        self.linear1 = nn.Linear(self.hidR, self.fc_hid_size)
        self.linear_mean = nn.Linear(self.fc_hid_size, args.prediction_window_size)
        self.linear_var = nn.Linear(self.fc_hid_size, args.prediction_window_size)

        if (self.hw > 0):
            self.highway = nn.Linear(self.hw * self.variables, 1)

        self.dropout = nn.Dropout(p=args.dropout)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        # batch_size * window * variables
        r = x.permute(1, 0, 2).contiguous()
        _,r=self.rnn1(r)
        r = self.dropout(torch.squeeze(r[-1:,:,:], 0))
        r = self.linear1(r)
        out = self.linear_mean(r)
        pre_var = self.linear_var(r)
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.view(-1, self.hw * self.variables)
            z = self.highway(z)
            z = z.view(-1, self.prediction_window_size)
            out = out + z
        if self.output is not None:
            out = self.output(out)
        return out

