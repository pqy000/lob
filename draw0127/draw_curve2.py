import matplotlib
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 6))

# plt.subplot(111)
num = 2
x = [1,2,3,4,5,6,7,8,9,10,11,12]#点的横坐标
k1   = [49.4,48.3,47.53,49.96,48.54,51.36,50.21,49.56,48.14,52.27,48.34,48.32]

plt.plot(x,k1,'s-',color = 'blue',label="CE-Sharp")
# plt.plot(x,k3,'h-',color = 'red', lw=2, label="CE-Sharp")
plt.xlabel("Month", fontdict={'family' : 'Times New Roman', 'size': 23})#横坐标名字
plt.ylabel("Accuracy(%)", fontdict={'family' : 'Times New Roman', 'size': 24})#纵坐标名字
plt.xticks(fontsize=25,fontproperties = 'Times New Roman' )
plt.yticks(fontsize=25,fontproperties = 'Times New Roman' )
# plt.legend(loc = "best")#图例
# plt.title("Air-quality")
plt.ylim(40, 60)
plt.legend(loc = "best", prop={'family' : 'Times New Roman', 'size': 19.5})
plt.grid()
fig.tight_layout()
ax1=plt.gca()
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
plt.savefig("accuracy127.pdf")
plt.show()
