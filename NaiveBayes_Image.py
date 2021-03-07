
from math import sqrt
from math import exp
from math import pi
import scipy.io
import numpy as np

# Yuxue Piao
# Student ID : 1214904015
# Project 1
#


# Compare predicted label with actual label and return accuracy percentage.
def calculateAccuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0



def separate_by_label(dataset):
    result = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in result:
            result[class_value] = list()
        result[class_value].append(vector)
    return result


# Calculate the mean, stdev and count for each column in a dataset
def maximumLikelihoodForGaussionDistribution(dataset):
    # after derivate the mu of formular is equal with mean of inputs.
    # after derivate the gaussion distribution function the sigma is equal with std
    summaries = [(np.mean(column), np.std(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries



def seperateTrainingDataByClassAndCalculateMLE(dataset):
    separated = separate_by_label(dataset)
    estimatedParmetersAndResult = dict()
    for label, rows in separated.items():
        estimatedParmetersAndResult[label] = maximumLikelihoodForGaussionDistribution(rows)
    return estimatedParmetersAndResult


# Gaussian probability distribution
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each image feature
def calculate_class_probabilities(summaries, eachImage):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(eachImage[i], mean, stdev)
    return probabilities


# Predict the class for a given feature
def predict(summaries, eachImage):
    probabilities = calculate_class_probabilities(summaries, eachImage)
    best_label, best_prob = None, -1
    for current_class_label, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = current_class_label
    return best_label


# Naive Bayes
def naive_bayes(train, test):
    summaries = seperateTrainingDataByClassAndCalculateMLE(train)
    predictions = list()
    for eachImage in test:
        output = predict(summaries, eachImage)
        predictions.append(output)
    return predictions



def doAnalysis():
    data = scipy.io.loadmat('fashion_mnist.mat')
    data_size = len(data['trX'])
    test_size = len(data['tsX'])
    TrainingData = []
    TestData = []

    for i in range(0, data_size):
        TrainingData.append([np.mean(data['trX'][i]), np.std(data['trX'][i]), data['trY'][0][i]])

    for i in range(0, test_size):
        TestData.append([np.mean(data['tsX'][i]), np.std(data['tsX'][i]), data['tsY'][0][i]])

    prediction = naive_bayes(TrainingData, TestData)

    print(prediction)
    actual_label = [row[-1] for row in TestData]
    print(actual_label)

    accuracy = calculateAccuracy(actual_label, prediction)

    print("The Accuracy is {} %".format(accuracy))
