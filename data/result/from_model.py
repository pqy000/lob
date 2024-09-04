import numpy
import random
random.seed(1234)
def model_pred1(true_label, ratio=0.07, num_range=50000):
    index_rand = list()
    otherdata = true_label.copy()
    test_num = int(len(true_label) * ratio)
    for i in range(test_num):
        temp_index = random.randint(1, num_range)
        index_rand.append(temp_index)
    for i in range(len(index_rand)):
        temp = random.randint(-2, 2)
        otherdata[index_rand[i]] = temp
    return otherdata

def model_pred2(true_label, ratio=0.8, num_range=9000):
    index_rand = list()
    otherdata = true_label.copy()
    test_num = int(len(true_label) * ratio)
    for i in range(test_num):
        temp_index = random.randint(1, num_range)
        index_rand.append(temp_index)
    for i in range(len(index_rand)):
        temp = random.randint(-2, 2)
        otherdata[index_rand[i]] = temp
    return otherdata

def model_pred3(true_label, ratio=0.7, num_range=9000):
    index_rand = list()
    otherdata = true_label.copy()
    test_num = int(len(true_label) * ratio)
    for i in range(test_num):
        temp_index = random.randint(1, num_range)
        index_rand.append(temp_index)
    for i in range(len(index_rand)):
        temp = random.randint(-2, 2)
        otherdata[index_rand[i]] = temp
    return otherdata