
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
from time import strftime, localtime

# 打印当前时间
def printTime():
    temp = strftime("%H:%M", localtime())
    return temp

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.Ck = args.CNN_kernel
        self.cnn = None

        self.hw = args.highway_window
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        # if (self.hw > 0):
        #     self.highway = nn.Linear(self.hw, 1)
        # self.output = None
        # if (args.output_fun == 'sigmoid'):
        #     self.output = F.sigmoid
        # if (args.output_fun == 'tanh'):
        #     self.output = F.tanh
        self.linear2 = nn.Linear(self.m, data.num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # CNN
        c = x.view(-1, 1, self.P, self.m)
        temp = self.conv1(c)
        c = F.relu(temp)
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN 
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)

        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        # if (self.skip > 0):
        #     self.pt=int(self.pt)
        #     s = c[:, :, int(-self.pt * self.skip):].contiguous()
        #
        #     s = s.view(batch_size, self.hidC, self.pt, self.skip)
        #     s = s.permute(2, 0, 3, 1).contiguous()
        #     s = s.view(self.pt, batch_size * self.skip, self.hidC)
        #     _, s = self.GRUskip(s)
        #     s = s.view(batch_size, self.skip * self.hidS)
        #     s = self.dropout(s)
        #     r = torch.cat((r, s), 1)

        res = self.linear1(r)
        res = F.relu(res)
        res = self.linear2(res)


        # if (self.hw > 0):
        #
        #     z = x[:, -self.hw:, :]
        #     z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
        #     z = self.highway(z)
        #     z = z.view(-1, self.m)
        #     res = res + z
        res = torch.softmax(res, dim=1)

        return res
