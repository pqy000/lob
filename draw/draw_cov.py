import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

# num = 2
# x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
# k1 = [0.0000,0.0000,0.000,0.000,0.0000,0.0000,0.0002,0.0003,0.0004,0.0007,0.0011]#线1的纵坐标
# k2 = [0.0000,0.0000,0.000,0.000,0.0001,0.0002,0.0004,0.0005,0.0009,0.0013,0.0022]#线2的纵坐标
# k3 = [0.0000,0.0000,0.000,0.0001,0.0008,0.0012,0.0014,0.0021,0.0023,0.0029,0.0038]

num = 2
x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1 = [0.0000,0.0000,0.000,0.000,0.0000,0.0000,0.02,0.03,0.04,0.07,0.11]#线1的纵坐标
k2 = [0.0000,0.0000,0.000,0.000,0.01,0.02,0.04,0.05,0.09,0.13,0.22]#线2的纵坐标
k3 = [0.0000,0.0000,0.000,0.02,0.08,0.12,0.14,0.21,0.23,0.29,0.38]

plt.plot(x,k1,'s-',color = 'blue',label="Ours")
plt.plot(x,k2,'*-',color = 'green',label="Multi-Evidential")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="MC-Dropout")
plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Distance between $\hat{\Sigma}$ and $\Sigma$", fontdict={'family' : 'Times New Roman', 'size': 22})#纵坐标名字
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
plt.savefig("Accuracy_cov.pdf")
plt.show()