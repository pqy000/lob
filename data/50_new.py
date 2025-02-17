import numpy as np
import random
from from_model import *
import pickle5 as pickle

word = pickle.load(open("TS_Data50.pkl", 'rb'), encoding='utf-8')
word = word.dropna()
raw_data = word['FCLOSE']
word,raw_time = word['FCLOSE'].to_numpy(), word['DATE'].to_numpy()
true_label = list()
threshold, threshold2 = 0.0004, 0.0002
window_size = 20
total_len = 1000
start_time = 256600
end_time = 317290
total_len = 317290 - 256600
length = 100

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

combined_array = np.column_stack((data[length:], true_label))
save_path = './50_2020'
np.save(save_path, combined_array)
