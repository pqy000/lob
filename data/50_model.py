import xlwt
import numpy as np
import random
from from_model import *
import xlwt
import pickle5 as pickle
word = pickle.load(open("TS_Data50.pkl", 'rb'), encoding='utf-8')
word = word.dropna()
raw_data = word['FCLOSE']
word,raw_time = word['FCLOSE'].to_numpy(), word['DATE'].to_numpy()
true_label = list()

threshold, threshold2 = 0.0006, 0.0003
window_size = 20
total_len = 1000
start_time = 256600
end_time = 317290 #2020
total_len = 317290 - 256600

data, time_data = word[start_time:end_time], raw_time[start_time:end_time]
for i in range(len(data) - 100):
    total = sum(data[i:i+window_size])/window_size
    ratio = float(data[i]) / float(total) - 1
    if ratio > threshold: temp = 2
    elif ratio <= threshold and ratio > threshold2: temp = 1
    elif ratio <= threshold2 and ratio > -threshold2: temp = 0
    elif ratio <= -threshold2 and ratio > -threshold: temp = -1
    elif ratio <= -threshold: temp = -2
    true_label.append(temp)

# Create a sample ndarray with shape (100,)
array_data = np.random.rand(100)

# Save the ndarray to a file
save_path = '/mnt/data/array_data.npy'
np.save(save_path, array_data)


workbook = xlwt.Workbook(encoding= 'ascii')
worksheet = workbook.add_sheet("My new Sheet")
worksheet.write(0, 0, "close价格")
worksheet.write(0, 1, "价格差距")
worksheet.write(0, 2, "真实标签")
worksheet.write(0, 3, "预测标签")
worksheet.write(0, 4, "原仓位")
worksheet.write(0, 5, "预测仓位")
worksheet.write(0, 6, "成本")
worksheet.write(0, 7, "S值")
worksheet.write(0, 8, "盈利")
worksheet.write(0, 9, "flag")
worksheet.write(0, 10, "cum_s")
worksheet.write(0, 11, "cum_win")


otherdat = model_pred1(true_label)
# otherdat2 = model_pred2(true_label)
# otherdat3 = model_pred3(true_label)

acc_list = list()
for i in range(len(otherdat)):
    if otherdat[i] == true_label[i]:
        acc_list.append(i)
s_value = list()
count = 0
count_s, count_win = 0, 0
for i in range(1, len(true_label)):
    worksheet.write(i, 0, data[i])
    worksheet.write(i, 1, data[i+1]-data[i])
    worksheet.write(i, 2, true_label[i])
    worksheet.write(i, 3, otherdat[i])
    worksheet.write(i, 4, float(true_label[i])/2)
    worksheet.write(i, 5, float(otherdat[i])/2)
    worksheet.write(i, 6, float(data[i+1]-data[i])*0.0002)
    # s = float(data[i+1]-data[i])/float(data[i])*otherdat[i]
    s = float(data[i + 1] - data[i]) / float(data[i]) * true_label[i]
    worksheet.write(i, 7, s)
    worksheet.write(i, 8, s - float(data[i+1]-data[i])*0.0002)
    count_s += s
    count_win += s - float(data[i+1]-data[i])*0.0002
    if s > -0.001 :
        s_value.append(s)
        count += 1
        worksheet.write(i, 9, "1")
    else:
        worksheet.write(i, 9, "0")
    # s_value.append(s)
    worksheet.write(i, 10, count_s)
    worksheet.write(i, 11, count_win)

avg_s = sum(s_value) / len(s_value)
std_s = np.std(s_value)
sharp_value = (avg_s / std_s) * np.sqrt(240)

print("#" * 100)
print("Original Accuracy: {:5.4f}".format(len(acc_list)/total_len))
print("mean_s: {:5.4f} std_s: {:5.4f} "
      "result_s: {:5.4f}".format(avg_s, std_s, sharp_value))

workbook.save("result_other/2020_50_result_"+str(window_size)+".xls")

