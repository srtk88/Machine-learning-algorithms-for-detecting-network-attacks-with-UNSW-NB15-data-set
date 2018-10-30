#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 18:13:26 2018

@author: Ke-Hsin, Lo
"""

import unicodecsv
import random
import operator
import math
import numpy

import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import matplotlib.pyplot as plt

# getdata() function definition
def getdata(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.reader(f)
        return list(reader)





def cosine_similarity(v1, v2):

    sum_xx, sum_xy, sum_yy = 0.0, 0.0, 0.0
 #   print "len: %d" %(len(v1))
    for i in range(0, len(v1)-1):
 #       print (v1[i])
        sum_xx += math.pow(float(v1[i]), 2)
        sum_xy += float(v1[i]) * float(v2[i])
        sum_yy += math.pow(float(v2[i]), 2)

    return sum_xy / math.sqrt(sum_xx * sum_yy)

def cosine_distance(v1, v2):
    1-cosine_similarity(v1,v2)

# KNN prediction and model training
def knn_predict(test_data, train_data, k_value, category):
    totalcount = 0
    for i in test_data: #select tested data
        cos_similarity_list = [] # all distance array

        classNum=dict() #a dictionary of nebor
        classNum['Normal'] = 0
        classNum['Reconnaissance'] = 0
        classNum['Exploits'] = 0
        classNum['Fuzzers'] = 0
        classNum['DoS'] = 0
        classNum['Generic'] = 0
        classNum['Shellcode'] = 0
        classNum['Analysis'] = 0
        classNum['Worms'] = 0
        classNum['Backdoors'] = 0

        jcount = 0

        for j in train_data: # find in train data to get the nearest point
       #     print "i: %s" %(i)
            cos_sm = cosine_similarity(i, j)  #  1 test data  train set
            cos_similarity_list.append((category[jcount], cos_sm)) #the distance with the category
#            print cos_similarity_list # just for debugging and observing; in general running, this line will not be used.
            print "count: %s" %(jcount)
            cos_similarity_list.sort(key=operator.itemgetter(1), reverse=True) #use cos piority
            ''' similarity priority list has been built; we can find the first k nearest neighbors '''
            jcount += 1
            totalcount += 1
            print "Processing: %s" % (totalcount)

        knn = cos_similarity_list[:k_value]  # select first k neighbors

        print knn
        for k in knn: #k[0] is the most simliar.
            if k[0] == 'Normal':
                classNum['Normal'] += 1
            elif k[0] == 'Reconnaissance':
                classNum['Reconnaissance'] += 1
            elif k[0] == 'Exploits':
                classNum['Exploits'] += 1
            elif k[0] == 'Fuzzers':
                classNum['Fuzzers'] += 1
            elif k[0] == 'DoS':
                classNum['DoS'] += 1
            elif k[0] == 'Generic':
                classNum['Generic'] += 1
            elif k[0] == 'Shellcode':
                classNum['Shellcode'] += 1
            elif k[0] == 'Analysis':
                classNum['Analysis'] += 1
            elif k[0] == 'Worms':
                classNum['Worms'] += 1
            elif k[0] == 'Backdoors':
                classNum['Backdoors'] += 1

    
#        print  "result: %d %d %d %d %d" %(classNum['Normal'],  classNum['Reconnaissance'], classNum['Exploits'], classNum['Fuzzers'], classNum['DoS'])
        max_value = max(classNum, key=classNum.get) # max(classNum)
        print "max_value %s" %(max_value)

    #    recover_key(classNum, max_value)

 #       max_index = recover_key(classNum, max_value)
 #       print "max_index %s" %(max_index)
        i.append(max_value) # append prediction; tag category

def recover_key(dictionary, value):
     for a_key in dictionary.keys():
         if (dictionary[a_key] == value):
             return a_key

# Accuracy calculation function
def accuracy(test_data, true_result):
    correct = 0
    for i in test_data:
        #print len(i)
        #print i[len(i)-1]

        jcount = 0
        if true_result[jcount] == i[len(i)-1]:
            correct += 1
            jcount+=1

    accuracy = float(correct) / len(test_data) * 100  # accuracy
    return accuracy


def KNN(K, train_x, train_y, test_x, test_y):
   # dataset = getdata('UNSW_NB15_training-set_selected.csv')  # getdata function call with csv file as parameter
#    print len(dataset)
 #   train_dataset, test_dataset = shuffle(dataset)  # train test data split
  #  K = 3  # Assumed K value

    train_dataset = train_x.tolist()
    print "Number of training X: %s" %len(train_dataset)
    print "Number of training Y: %s" %len(train_y)
    test_dataset = test_x.tolist()
    print "Number of testing X: %s" %len(test_dataset)

    print "Training Set KNN Process:"
    knn_predict(train_dataset, train_dataset, K, train_y)
    print "Testing Set KNN Process:"
    knn_predict(test_dataset, train_dataset, K, train_y)
    atrain = round(accuracy(train_dataset, train_y),5)
    TrainError = float(100.00000- float(atrain))
    atest = round(accuracy(test_dataset, test_y),5)
    TestError = 100.00000- atest
    # print test_dataset
    print "Accuracy of train_dataset : ", atrain
    print "Train error : ", TrainError
    print "Accuracy of test_dataset: ", atest
    print "Test error: ", TestError
    return TrainError, TestError, atrain, atest

def load_train_data(train_ratio=0.12):
    data = pd.read_csv('./UNSW_NB15_training-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(33)] + ['logic']+['y'])
    Xt = numpy.asarray(data[['x%i' % (i) for i in range(33)]])
    yt = numpy.asarray(data['y'])

    return sklearn.model_selection.train_test_split(Xt, yt, test_size=1 - train_ratio, random_state=0)


def load_test_data(train_ratio=0.88):
    data = pd.read_csv('./UNSW_NB15_testing-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(33)] + ['logic']+['y'])
    Xtt = numpy.asarray(data[['x%i' % (i) for i in range(33)]])
    ytt = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(Xtt, ytt, test_size=1 - train_ratio, random_state=0)

def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train
                                                                                                )))  # Transforms features by scaling each feature to a given range(0~1) in order to reinforce dataset and fit training set.
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale

"""preprocessing x and y of training data"""
x_train2, t1, y_train, t2 = load_train_data(train_ratio=0.003) #1
"""preprocessing x and y of testing data"""
t3, X_test, t4, y_test = load_test_data(train_ratio=(1-0.003)) #2

"""scale X dataset"""
X_train_scale, X_test_scale = scale_features(x_train2, X_test, 0, 1)
TrainError = []
TestError = []
TrainAccuracy = []
TestAccuracy = []
plt.figure(2)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
plt.figure(3)
bx1 = plt.subplot(311)
bx2 = plt.subplot(312)
x = []

'''knn start: for small sample, start from 1; this from 9 is for this big sample set. Because there are same similarity in diffrent kind.'''
for k in range(13,3,-1): #3
    print "K: %d" %(k)
    TrainErrorTemp, TestErrorTemp, AoTrain, AoTest = KNN(k, x_train2, y_train, X_test, y_test)
    TrainError.append(TrainErrorTemp)
    TestError.append(TestErrorTemp)
    TrainAccuracy.append(AoTrain)
    TestAccuracy.append( AoTest)
    print " "
    x.append(k)


plt.sca(ax1)
plt.plot(x, TrainError)


plt.sca(ax2)
plt.plot(x, TestError)


plt.sca(bx1)
plt.plot(x, TrainAccuracy)


plt.sca(bx2)
plt.plot(x, TestAccuracy)


plt.xlabel('x axis') # make axis labels
plt.ylabel('y axis')
plt.show()

