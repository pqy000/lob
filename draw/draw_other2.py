import matplotlib.pyplot as plt
import numpy as np

# name_list = ["SegNet", "FCN8", "Bayes-Seg", "Eigen", "MCDropout", "Ours"]
# x = np.arange(len(name_list))
# num_list = [66.1, 61.8, 68, 65.6, 70.6, 72.3]
# plt.bar(range(len(num_list)), num_list, width=0.48, color="blue", tick_label=name_list)
# plt.ylabel('Accuracy', fontdict={'family' : 'Times New Roman', 'size': 12})
# plt.xticks(x, labels=name_list, fontproperties = 'Times New Roman', size=12)
# plt.yticks(fontproperties = 'Times New Roman', size=12)
# ax1=plt.gca()
# plt.ylim(57, 75)
# ax1.spines['top'].set_linewidth('1.3')
# ax1.spines['bottom'].set_linewidth('1.3')
# ax1.spines['left'].set_linewidth('1.3')
# ax1.spines['right'].set_linewidth('1.3')
# plt.savefig("1.pdf")
# plt.show()


name_list = ["SegNet", "FCN8", "Bayes-Seg", "Eigen", "MCDropout", "Ours"]
x = np.arange(len(name_list))
num_list = [-8.302,-8.093,-9.143,-9.404,-9.988,-10.677]
plt.bar(range(len(num_list)), num_list, width=0.48, color="blue", tick_label=name_list)
plt.ylabel('Negative Log-Likelihood', fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xticks(x, labels=name_list, fontproperties = 'Times New Roman', size=12)
plt.yticks(fontproperties = 'Times New Roman', size=12)
ax1=plt.gca()
plt.ylim(-7, -11.5)
ax1.spines['top'].set_linewidth('1.3')
ax1.spines['bottom'].set_linewidth('1.3')
ax1.spines['left'].set_linewidth('1.3')
ax1.spines['right'].set_linewidth('1.3')
plt.savefig("2.pdf")
plt.show()


