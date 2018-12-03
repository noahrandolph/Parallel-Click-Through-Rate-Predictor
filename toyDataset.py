#!/usr/bin/env python

import numpy as np
import csv

SEED = 2615
NUMERICCOLS = 2
ONEHOTCOLS = 2


# start Spark Session
from pyspark.sql import SparkSession
app_name = "loadAndEDA"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext


def generateToyDataset(w=[8, -3, -1, 3, 8]):
    '''generate toy logistic regression dataset with numerical and 1-hot encoded features'''
    nrows=8
    np.random.seed(SEED)
    x1 = np.random.randint(0, 10, nrows)
    x2 = np.random.randint(0, 10, nrows)
    x3 = np.random.randint(0, 2, nrows) # simulate 1-hot
    x4 = np.ones(nrows, np.int8) - x3   # with x3 and x4
    noise = np.random.normal(5, 1, nrows)
    v = (w[0] + x1*w[1] + x2*w[2] + x3*w[3] + x4*w[4] + noise)
    y = (v>0) * 2 - 1 # y = 1 or -1
    df = spark.createDataFrame(zip(y.tolist(), x1.tolist(), x2.tolist(), x3.tolist(), x4.tolist()))
    oldColNames = df.columns
    newColNames = ['Label']+['I{}'.format(i) for i in range(0,2)]+['C{}'.format(i) for i in range(0,2)]
    for oldName, newName in zip(oldColNames, newColNames):
        df = df.withColumnRenamed(oldName, newName)
    return df


def logLoss(dataRDD, W):
    """
    Compute mean squared error.
    Args:
        dataRDD - each record is a tuple of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    """
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1]))
    ################## YOUR CODE HERE ##################
    loss = augmentedData.map(lambda p: (np.log(1 + np.exp(-p[1] * np.dot(W, p[0]))))) \
                        .reduce(lambda a, b: a + b)
    ################## (END) YOUR CODE ##################
    return loss


def GDUpdate(dataRDD, W, learningRate = 0.1):
    """
    Perform one OLS gradient descent step/update.
    Args:
        dataRDD - records are tuples of (features_array, y)
        W       - (array) model coefficients with bias at index 0
    Returns:
        new_model - (array) updated coefficients, bias at index 0
    """
    # add a bias 'feature' of 1 at index 0
    augmentedData = dataRDD.map(lambda x: (np.append([1.0], x[0]), x[1])).cache()
    
    ################## YOUR CODE HERE ################# 
    grad = augmentedData.map(lambda p: (-p[1] * (1 - (1 / (1 + np.exp(-p[1] * np.dot(W, p[0]))))) * p[0])) \
                        .reduce(lambda a, b: a + b)
    new_model = W - learningRate * grad
    ################## (END) YOUR CODE ################# 
    
    return new_model


def dfToRDD(row):
    '''
    Converts dataframe row to rdd format.
        From: DataFrame['Label', 'I0', ..., 'C0', ...]
        To:   (features_array, y)
    '''
#     fields = np.array(line.split(';'), dtype = 'float')
#     features,quality = fields[:-1], fields[-1]
    
    features_list = [row['I{}'.format(i)] for i in range(0, NUMERICCOLS)] + [row['I{}'.format(i)] for i in range(0, ONEHOTCOLS)]
    features_array = np.array(features_list)
    y = row['Label']
    return (features_array, y)


def normalize(dataRDD):
    """
    Scale and center data around the mean of each feature.
    """
    featureMeans = dataRDD.map(lambda x: x[0]).mean()
    featureStdev = np.sqrt(dataRDD.map(lambda x: x[0]).variance())
    normedRDD = dataRDD.map(lambda x: ((x[0] - featureMeans)/featureStdev, x[1]))
    return normedRDD


# create a toy dataset that includes 1-hot columns for development
df = generateToyDataset()   

# convert dataframe to RDD for homegrown logistic regression
trainRDD = df.rdd.map(dfToRDD)

# normalize RDD
normedRDDcached = normalize(trainRDD).cache()
print(normedRDDcached.take(1))

# create initial weights to train
featureLen = len(normedRDDcached.take(1)[0][0])
wInitial = np.random.normal(size=featureLen+1) # add 1 for bias

# 1 iteration of gradient descent
w = GDUpdate(normedRDDcached, wInitial)

nSteps = 5
for idx in range(nSteps):
    print("----------")
    print(f"STEP: {idx+1}")
    w = GDUpdate(normedRDDcached, w)
    loss = logLoss(normedRDDcached, w)
    print(f"Loss: {loss}")
    print(f"Model: {[round(i,3) for i in w]}")