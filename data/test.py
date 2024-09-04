import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

# plt.subplot(111)
num = 2
x = [16,32,48,64,80,96,128,144]#点的横坐标
k1   = [23.1,27.33,39.28,44.32,42.31,52.3,49.4,48.3]
k2   = [35.42,44.33,56.58,53.32,53.31,54.3,54.1,55.3]
k3   = [24.42,31.33,42.58,46.32,49.31,56.3,52.4,51.3]

plt.plot(x,k1,'s-',color = 'blue',label="C-Entropy")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="CE-Sharp")
plt.plot(x,k2,'*-',color = 'green', lw=2, label="CE-Sharp(GPU)")
plt.xlabel("Window size", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Accuracy(%)", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
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
plt.ylim(21, 60)
plt.savefig("Accuracy1.pdf")
plt.show()

fig = plt.figure(figsize=(8, 6))
num = 2
x = [16,32,48,64,80,96,128,144]#点的横坐标
k1   = [1.4084,1.6482,1.8446,1.928,2.0631,2.3321,2.2732,2.25]
k2   = [1.4284,2.1582,2.2946,2.3928,2.36,2.4121,2.3432,2.295]
k3   = [1.3284,1.582,1.6646,1.728,1.76,2.0321,1.9132,1.95]

plt.plot(x,k1,'s-',color = 'blue',label="C-Entropy")
plt.plot(x,k3,'h-',color = 'red', lw=2, label="CE-Sharp")
plt.plot(x,k2,'*-',color = 'green', lw=2, label="CE-Sharp(GPU)")
plt.xlabel("Window size", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("S-value(%)", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.legend(loc = "best")#图例
# plt.title("Air-quality")
# plt.ylim(3, 18)
plt.legend(loc = "best", prop={'family' : 'Times New Roman', 'size': 19.5})
plt.grid()
# fig.tight_layout()
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
plt.ylim(1.0, 2.5)
plt.savefig("Accuracy1.pdf")
plt.show()