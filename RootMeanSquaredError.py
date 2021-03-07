import math

def rootMeanSquaredError(predictedVector, actualVector):
    sum_of_error = 0.0
    for i in range(len(predictedVector)):
        sum_of_error += math.pow( predictedVector[i] - actualVector[i] , 2)

    return math.sqrt(sum_of_error/float(len(predictedVector)))



x = [0, 1, 2, 3, 4]
y = [0.1, 0.9, 2.1, 2.9, 4.1]

y_1 = [0, 1, 2, 3, 4]
y_2 = [0.1, 0.9, 2.1, 2.9, 4.1]


print(rootMeanSquaredError(y_1, y))
print(rootMeanSquaredError(y_2, y))

