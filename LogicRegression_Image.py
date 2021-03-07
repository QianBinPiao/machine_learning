import scipy.io
from math import exp
import numpy as np

# Yuxue Piao
# Student ID : 1214904015
# Project 1
#

# beta0 + beta1*x1 + beta2 * x2
def predict(features, coefficients):
    yhat = coefficients[0]
    for i in range(len(features)-1):
        yhat += coefficients[i + 1] * features[i]
    return 1.0 / (1.0 + exp(-yhat))


def Gradient_Ascent_Algorithm(train_features, learning_rate, epoch):
    coefficients = np.random.random_sample((3,))
    for i in range(epoch):
        for eachRow in train_features:
            yhat = predict(eachRow, coefficients)
            error = eachRow[-1] - yhat
            coefficients[0] = coefficients[0] + learning_rate * error * yhat * (1.0 - yhat)
            for j in range(len(eachRow) - 1):
                coefficients[j + 1] = coefficients[j + 1] + learning_rate * error * yhat * (1.0 - yhat) * eachRow[j]
    return coefficients


def logisticRregression(train_feature, test_feature, learning_rate, epoch):
    predicted_result = list()
    coefficients = Gradient_Ascent_Algorithm(train_feature, learning_rate, epoch)
    for eachImage in test_feature:
        yhat = predict(eachImage, coefficients)
        predicted_result.append(round(yhat))
    return predicted_result


def calculateAccuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def doAnalysis():

    data = scipy.io.loadmat('fashion_mnist.mat')
    data_size = len(data['trX'])
    test_size = len(data['tsX'])
    Training_features = []
    Test_features = []

    for i in range(0, data_size):
        Training_features.append([np.mean(data['trX'][i]), np.std(data['trX'][i]), data['trY'][0][i]])

    for i in range(0, test_size):
        Test_features.append([np.mean(data['tsX'][i]), np.std(data['tsX'][i]), data['tsY'][0][i]])


    predicted_result = logisticRregression(Training_features, Test_features, 0.1, 6000)

    actual_label = [row[-1] for row in Test_features]

    print(actual_label)

    print(predicted_result)

    accuracy = calculateAccuracy(actual_label, predicted_result)
    print("The Accuracy is {} %".format(accuracy))
