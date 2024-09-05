import numpy as np
import random
import pickle5 as pickle

word = pickle.load(open("data/TS_Data300.pkl", 'rb'), encoding='utf-8')
word = word.dropna()
raw_data = word['FCLOSE']
feature = word[['OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOL',
                'FOPEN','FCLOSE','FHIGH','FLOW','FVOL']].values.astype(float)
word, raw_time = word['FCLOSE'].to_numpy(), word['DATE'].to_numpy()
true_label = list()
threshold, threshold2, window_size = 0.0004, 0.0002, 10
total_len = 1000
start_time = 256600
end_time = 317290
total_len = 317290 - 256600
length = 100

data, time_data = word[start_time:end_time], raw_time[start_time:end_time]
feature = feature[start_time:end_time]
for i in range(len(data)):
    total = sum(data[i:i+window_size])/window_size
    ratio = float(data[i]) / float(total) - 1
    if ratio > threshold: temp = 2
    elif ratio <= threshold and ratio > threshold2: temp = 1
    elif ratio <= threshold2 and ratio > -threshold2: temp = 0
    elif ratio <= -threshold2 and ratio > -threshold: temp = -1
    elif ratio <= -threshold: temp = -2
    true_label.append(temp)

true_label = np.expand_dims(true_label, axis=1)
combined_array = np.concatenate((feature, true_label), axis=1)
print(combined_array.shape)
# combined_array = np.column_stack((data[length:], true_label))
save_path = './newdata/300_2020.npy'
np.save(save_path, combined_array)
