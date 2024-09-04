import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

# plt.subplot(111)
num = 2
x = [16,32,48,64,80,96,128,144]#点的横坐标
k1   = [18.1,27.33,39.28,46.32,48.31,49.3,49.4,43.3]
k3   = [21.42,29.38,45.58,47.32,51.31,54.3,52.4,51.3]

# k1 = [15.474,16.18,16.429,16.221,16.682,16.873,16.673,17.048,17.859,16.955,16.882,16.968,16.942,15.944]#线1的纵坐标
# k2 = [14.615,15.457,17.634,18.153,18.533,19.263,18.256,18.783,19.49,17.301,18.417,19.457,19.591,19.575]#线2的纵坐标
# k3 = [7.07,10.07,11.4,13.9,14.8,15.1,14.18,15.81,14.04,13.07,13.18,13.49,13.37,13.9]
plt.plot(x,k1,'s-',color = 'blue',label="C-Entropy")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="CE-Sharp")
plt.xlabel("Window size", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Accuracy(%)", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.legend(loc = "best")#图例
# plt.title("Air-quality")
plt.ylim(13, 60)
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
