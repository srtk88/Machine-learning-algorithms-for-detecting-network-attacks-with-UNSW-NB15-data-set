#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed July 18 11:04:32 2018

@author: Ke-Hsin,Lo
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import nesssary package
import math
import sys
import numpy
from numpy import Inf
import pandas as pd
import sklearn.metrics
import sklearn.model_selection
import sklearn.linear_model
import sklearn.preprocessing
import random
import matplotlib.pyplot as plt

def load_train_data(train_ratio=1):
    data = pd.read_csv('./UNSW_NB15_training-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(37)] + ['y'])
    Xt = numpy.asarray(data[['x%i' % (i) for i in range(37)]])
    yt = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(Xt, yt, test_size=1 - train_ratio, random_state=0)


def load_test_data(train_ratio=0):
    data = pd.read_csv('./UNSW_NB15_testing-set_selected.csv', header=None,
                       names=['x%i' % (i) for i in range(37)] + ['y'])
    Xtt = numpy.asarray(data[['x%i' % (i) for i in range(37)]])
    ytt = numpy.asarray(data['y'])
    return sklearn.model_selection.train_test_split(Xtt, ytt, test_size=1 - train_ratio, random_state=0)


def scale_features(X_train, X_test, low=0, upp=1):
    minmax_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(low, upp)).fit(numpy.vstack((X_train)))  # Transforms features by scaling each feature to a given range(0~1) in order to reinforce dataset and fit training set.
    X_train_scale = minmax_scaler.transform(X_train)
    X_test_scale = minmax_scaler.transform(X_test)
    return X_train_scale, X_test_scale


def logistic(x):
    return 1.0 / (1 + math.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


def logistic_log_likelihood_i(x_i, y_i, theta):  # 0/1 : logL= y * logf + (1-y) * log(1-f)
    if y_i == 1.0:
        return math.log(logistic(numpy.dot(x_i, theta)))
    else:
        return math.log(1 - logistic(numpy.dot(x_i, theta)))


def logistic_log_likelihood(x, y, beta):
    return sum(logistic_log_likelihood_i(x_i, y_i, beta)
               for x_i, y_i in zip(x, y))


"""i is the index of the data point;
   j the index of the derivative"""


def logistic_log_partial_ij(x_i, yi, theta, j):    #calculate gives the gradient

    return (yi - logistic(numpy.dot(x_i, theta))) * x_i[j]

    """the gradient of the log likelihood
    corresponding to the i-th data point"""


def logistic_log_gradient_i(xi, yi, theta):   #calcaulate its it partial derivative by treating it as a function of just its ith variable, holding the o ther variable fixed
    return [logistic_log_partial_ij(xi, yi, theta, j)
            for j, _ in enumerate(theta)]


def logistic_log_gradient(x, y, beta):
    return reduce(vector_add,
                  [logistic_log_gradient_i(x_i, y_i, beta)
                   for x_i, y_i in zip(x, y)])


"""adds two vectors"""


def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip(v, w)]


"""scalar number multiplies vector ver 2; same as ver 1"""


def scalar_multiply_2(c, v):
    row = []

    row = numpy.asarray(c) * v

    return row


def error(xi, yi, theta):
    return yi - predict_prob(xi, theta)


"""evaluated error **2"""


def squared_error(xi, yi, theta):
    return error(xi, yi, theta) ** 2


"""the gradient corresponding to the ith squared error term"""


def squared_error_gradient(xi, yi, theta):
    return [-2 * x_ij * error(xi, yi, theta)
            for x_ij in xi]


""" calculate ridge penalty"""


def ridge_penalty(lamda, theta):
    return lamda * numpy.dot(theta[1:], theta[1:]) / 2


"""calculate ridge gradient simply"""


def ridge_penalty_gradient(lamda, theta):
    return [0] + [lamda * thetai for thetai in theta[1:]]


def logreg_sgd(X, y, alpha=.001, iters=100000, eps=1e-2, lamda=0.001):
    n, d = X.shape
  #  print(n, d)
    theta = numpy.zeros((d, 1))

    random.seed(0)
    theta = [random.random() for xi in X[0]]

    gradient_fn = logistic_log_gradient_i
    target_fn = logistic_log_likelihood_i  # target is to maximize likelihood value (approaching to zero)

    data = zip(X, y)

    alpha_0 = alpha  # a step length
    max_theta, max_value = -Inf, -500000
    counter_of_no_improve = 0  # counter
    while counter_of_no_improve < iters:

        log_likelihood_value = sum((target_fn(x_i, y_i, theta) + ridge_penalty(lamda, theta)) for x_i, y_i in
                                   data) / n  # According to theory of logistic likelihood; add ridge_penalty to prevent from overfitting.
        print(log_likelihood_value, max_value, max_theta, theta)  # print for processing verbosely
        if log_likelihood_value > max_value:  # if value bigger, it was improved.
            print("Likelihood Improved.")
            if abs(log_likelihood_value - max_value) < eps:  # once training finished, response the maximum theta.
                print("Target Minimum eps Achieved( < 1e-2 ): ", abs(log_likelihood_value - max_value))
                max_theta, max_value = theta, log_likelihood_value
                return max_theta
            else:
                print("eps: ", abs(log_likelihood_value - max_value))  # if not smaller than eps, continue training.

                """if find a new maximum, renew the value, and initialize the alpha, which is the walking length."""
            max_theta, max_value = theta, log_likelihood_value  # save the newest theta as max_theta for return the output and further training
            counter_of_no_improve = 0
            alpha = alpha_0
        else:
            """if it was not improved, narrow the walking length and try to walk next step(shrink the step size)."""
            counter_of_no_improve += 1
            print("Not improved. iter of Narrow the Step Length: ", counter_of_no_improve)
            alpha *= 0.9


        for xi, yi in data:
           gradient_i = gradient_fn(xi, yi, theta) + ridge_penalty_gradient(lamda, theta)  # calculate gradient

           theta = vector_add(theta, scalar_multiply_2(alpha, gradient_i))  # take a step

    # if training so many time and over the iterator number, finish training.
    theta = max_theta

    return theta


def predict_prob(X, theta):  # According to theory of logistic likelihood: probability
    return 1. / (1 + numpy.exp(-numpy.dot(X, theta)))


def evaluate(y_test, y_prob):  # Evaluation, in accordance with theory of statics.
    tpr = []
    fpr = []
    tp, fp, fn, tn = 0, 0, 0, 0  # true positive, false positive, false negative, true negative.
    for index, i in enumerate(y_test):
        j = index

     #   print("y_prob:",y_prob[j])
        round_prob=round(y_prob[j])
        if (i == 1 and round_prob == 1):
            tp = tp + 1
        elif (i == 0 and round_prob == 1):
            fp = fp + 1
        elif (i == 1 and round_prob == 0):
            fn = fn + 1
        elif (i == 0 and round_prob == 0):
            tn = tn + 1

    # accuracy
    correct = tp + tn
    total = tp + fp + fn + tn
    accuracy = correct / total

    # precision
    precision = tp / (tp + fp)

    # recall
    recall = tp / (tp + fn)

    # f1_score
    p = precision
    r = recall

    f1score = 2 * p * r / (p + r)

    print("Accuracy: {0}".format(accuracy))
    print("Precision: {0}".format(precision))
    print("Recall: {0}".format(recall))
    print("F1 Score: {0}".format(f1score))



def plot_roc_curve(y_test, y_prob):
    # compute tpr and fpr of different thresholds
    tpr = []
    fpr = []
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title('ROC ')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal', adjustable='box')
    fpr, tpr, thh = sklearn.metrics.roc_curve(y_test, y_prob, 1)
    plt.plot(fpr, tpr, color='green', marker='o', linestyle='solid')
    plt.savefig("roc_curve.png")
    plt.show()

def main(argv):
    """data preprocessing"""

    """preprocessing x and y of training data"""
    x_train2, t1, y_train, t2 = load_train_data(train_ratio=0.99)
    """preprocessing x and y of testing data"""
    t3, X_test, t4, y_test = load_test_data(train_ratio=0.01)
    """scale X dataset"""
    X_train_scale, X_test_scale = scale_features(x_train2, X_test, 0, 1)

    """training and get model"""
    theta = logreg_sgd(X_train_scale, y_train)

    """result output"""
    y_prob = predict_prob(X_train_scale, theta)
    print("Logreg train accuracy: %f" % (sklearn.metrics.accuracy_score(y_train, y_prob > .5)))
    y_prob = predict_prob(X_test_scale, theta)
    print("Logreg test accuracy: %f" % (sklearn.metrics.accuracy_score(y_test, y_prob > .5)))

    evaluate(y_test.flatten(), y_prob.flatten())
    plot_roc_curve(y_test.flatten(), y_prob.flatten())

if __name__ == "__main__":
    main(sys.argv)
