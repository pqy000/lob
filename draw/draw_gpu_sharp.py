import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random

fig, axs = plt.subplots(3, 4, figsize=(12, 10))
plt.subplots_adjust(left=0.125, bottom=0.125, right=0.9, top=0.9,
                wspace=0.5, hspace=0.3)
# axs.set_ylabel('S-Value')
# axs.set_xlabel('month')
lenx, leny = 3, 4
x = [1,2,3,4,5,6,7,8,9,10]#点的横坐标
month = 1
# plt.xlabel("Month", fontdict={'family': 'Times New Roman', 'size': 12})  # 横坐标名字
# plt.ylabel("S-Value(%)", fontdict={'family': 'Times New Roman', 'size': 12})  # 纵坐标名字
# plt.xticks(fontsize=25, fontproperties='Times New Roman')
# plt.yticks(fontsize=25, fontproperties='Times New Roman')
# plt.title(str(month))
# plt.ylim(0, 3)
# plt.legend(loc="best", prop={'family': 'Times New Roman', 'size': 12})
# plt.grid()
# axs.tick_params(axis='both', which='major', labelsize=10)

for i in range(lenx):
    for j in range(leny):
        k1 = [0.92084, 1.182, 1.3446, 2.028, 2.2631, 2.3421, 2.3332, 2.19, 2.18, 2.03]
        for k in range(len(k1)):
            k1[i] += random.random() / 7.0
        axs[i,j].plot(x,k1,'s-',color = 'orange',label="month"+str(month))
        axs[i,j].set_ylim(0.5, 3.5)
        axs[i, j].set_ylabel('Svalue', fontdict={'family': 'Times New Roman', 'size': 15})
        axs[i, j].set_xlabel('Month', fontdict={'family': 'Times New Roman', 'size': 15})
        axs[i, j].legend()
        month += 1
        # axs[i,j].set_xticks(fontsize=10, fontproperties='Times New Roman')
        # axs[i,j].set_yticks(fontsize=10, fontproperties='Times New Roman')
        # ax1=plt.gca()
        # ax1.spines['top'].set_linewidth('1.3')
        # ax1.spines['bottom'].set_linewidth('1.3')
        # ax1.spines['left'].set_linewidth('1.3')
        # ax1.spines['right'].set_linewidth('1.3')
plt.savefig("../gpu_svalue.pdf")
plt.show()
