import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

# plt.subplot(111)
num = 2
x = [2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1   =  [0,0,0,0,0,0,0.0001,0.0002,0.0004,0.0007,0.0011]
k2 = [0,0,0,0,0.0001,0.0002,0.0004,0.0005,0.0009,0.0013,0.0022]
k3   = [0,0,0,0.0001,0.0008,0.0011,0.0014,0.0021,0.0023,0.0029,0.0037]

# k1 = [15.474,16.18,16.429,16.221,16.682,16.873,16.673,17.048,17.859,16.955,16.882,16.968,16.942,15.944]#线1的纵坐标
# k2 = [14.615,15.457,17.634,18.153,18.533,19.263,18.256,18.783,19.49,17.301,18.417,19.457,19.591,19.575]#线2的纵坐标
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]
plt.plot(x,k1,'s-',color = 'blue',label="Ours")
plt.plot(x,k2,'s-',label="Multi-Evid")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="MC-Dropout")
plt.xlabel("Dimension", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Cov", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
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
plt.savefig("Accuracy1.pdf")
plt.show()
