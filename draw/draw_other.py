import matplotlib.pyplot as plt

x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1 = [0.0001,0.0003,0.0011,0.0021,0.0039,0.0055,0.0058,0.0077,0.0082,0.0094,0.011]#线1的纵坐标
k2 = [0.0001,0.0004,0.0027,0.0045,0.0079,0.0097,0.013,0.016,0.021,0.028,0.031]#线2的纵坐标
k3 = [0.0003,0.0004,0.0027,0.0045,0.0093,0.012,0.018,0.021,0.032,0.043,0.053]
plt.plot(x,k1,'s-',color = 'r',label="Ours")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Multi-Evidential")#o-:圆形
plt.plot(x,k3,'*-',label="MC-Dropout")
plt.xlabel("Dimension")#横坐标名字
plt.ylabel("RMSE")#纵坐标名字
plt.legend(loc = "best")#图例

plt.show()

x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1 = [0.011,0.0094,0.0082,0.0077,0.0058,0.0055,0.0039,0.0021,0.0011,0.0003,0.0001]#线1的纵坐标
k2 = [0.031,0.028,0.021,0.016,0.013,0.0097,0.0079,0.0045,0.0027,0.0004,0.0001]#线2的纵坐标
k3 = [0.053,0.043,0.032,0.021,0.018,0.012,0.0093,0.0045,0.0027,0.0004,0.0003]

plt.plot(x,k1,'s-',color = 'r',label="Ours")#s-:方形
plt.plot(x,k2,'o-',color = 'g',label="Multi-Evidence")#o-:圆形
plt.plot(x,k3,'*-',label="MC-Dropout")

plt.xlabel("Dimension")#横坐标名字
plt.ylabel("CORR")#纵坐标名字
plt.legend(loc = "best")#图例

plt.show()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

num = 2
x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1 = [0.0001,0.0003,0.0011,0.0021,0.0039,0.0055,0.0058,0.0077,0.0082,0.0094,0.011]#线1的纵坐标
k2 = [0.0001,0.0004,0.0027,0.0045,0.0079,0.0097,0.013,0.016,0.021,0.028,0.031]#线2的纵坐标
k3 = [0.0003,0.0004,0.0027,0.0045,0.0093,0.012,0.018,0.021,0.032,0.043,0.053]

plt.plot(x,k1,'s-',color = 'blue',label="Ours")
plt.plot(x,k2,'*-',color = 'green',label="Multi-Evidential")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="MC-Dropout")
plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Distance between $\hat{y}$ and $y$", fontdict={'family' : 'Times New Roman', 'size': 22})#纵坐标名字
plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.legend(loc = "best")#图例
# plt.title("Air-quality")
# plt.ylim(3, 18)
plt.legend(loc = "best", prop={'family' : 'Times New Roman', 'size': 19.5})
plt.grid()
fig.tight_layout()
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
plt.savefig("Accuracy123.pdf")
plt.show()



# fig = plt.figure(figsize=(8, 6))
#
# num = 2
# x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
# k3 = [0.98,0.97,0.95,0.936,0.89,0.867,0.823,0.79,0.77,0.71,0.67]#线1的纵坐标
# k2 = [1.00,1,0.98,0.96,0.95,0.92,0.89,0.865,0.843,0.83,0.79]#线2的纵坐标
# k1 = [1.00,1,1,1,0.99,0.97,0.95,0.92,0.91,0.87,0.85]
#
# plt.plot(x,k1,'s-',color = 'blue',label="Ours")
# plt.plot(x,k2,'*-',color = 'green',label="Multi-Evidential")
# plt.plot(x,k3,'h-',color = 'red', lw=2, label="MC-Dropout")
# plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
# plt.ylabel("CORR between $y$ and $\hat{y}$", fontdict={'family' : 'Times New Roman', 'size': 22})#纵坐标名字
# plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# # plt.legend(loc = "best")#图例
# # plt.title("Air-quality")
# # plt.ylim(3, 18)
# plt.legend(loc = "best", prop={'family' : 'Times New Roman', 'size': 19.5})
# plt.grid()
# fig.tight_layout()
# ax1=plt.gca()
# ax1.spines['top'].set_linewidth('1.3')
# ax1.spines['bottom'].set_linewidth('1.3')
# ax1.spines['left'].set_linewidth('1.3')
# ax1.spines['right'].set_linewidth('1.3')
# plt.savefig("Accuracy1.pdf")
# plt.show()


# fig = plt.figure(figsize=(8, 6))
#
# num = 2
# x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
# k1 = [0.011,0.0094,0.0082,0.0077,0.0058,0.0055,0.0039,0.0021,0.0011,0.0003,0.0001]#线1的纵坐标
# k2 = [0.031,0.028,0.021,0.016,0.013,0.0097,0.0079,0.0045,0.0027,0.0004,0.0001]#线2的纵坐标
# k3 = [0.053,0.043,0.032,0.021,0.018,0.012,0.0093,0.0045,0.0027,0.0004,0.0003]
#
# plt.plot(x,k1,'s-',color = 'blue',label="MC-Dropout")
# plt.plot(x,k2,'*-',color = 'green',label="Multi-Evidence")
# plt.plot(x,k3,'h-',color = 'red', lw=2, label="Ours")
# plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
# plt.ylabel("CORR", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
# plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# # plt.legend(loc = "best")#图例
# # plt.title("Air-quality")
# # plt.ylim(3, 18)
# plt.legend(loc = "best", prop={'family' : 'Times New Roman', 'size': 19.5})
# plt.grid()
# fig.tight_layout()
# ax1=plt.gca()
# ax1.spines['top'].set_linewidth('1.3')
# ax1.spines['bottom'].set_linewidth('1.3')
# ax1.spines['left'].set_linewidth('1.3')
# ax1.spines['right'].set_linewidth('1.3')
# plt.savefig("Accuracy1.pdf")
# plt.show()