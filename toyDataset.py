#!/usr/bin/env python

import numpy as np
import csv

SEED = 2615


# start Spark Session
from pyspark.sql import SparkSession
app_name = "loadAndEDA"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext


def generateToyDataset(w=[0, 0, 0, 0, 0]):
    '''generate toy logistic regression dataset with numerical and 1-hot encoded features'''
    size=8
    np.random.seed(SEED)
    x1 = np.random.randint(0, 10, size)
    x2 = np.random.randint(0, 10, size)
    x3 = np.random.randint(0, 2, size) # simulate 1-hot
    x4 = np.ones(size, np.int8) - x3   # with x3 and x4
    noise = np.random.normal(5,1,size)
    v = (w[0] + x1*w[1] + x2*w[2] + x3*w[3] + x4*w[4] + noise)
    y = (v>0).astype(int)
    df = spark.createDataFrame(zip(y.tolist(), x1.tolist(), x2.tolist(), x3.tolist(), x4.tolist()))
    oldColNames = df.columns
    newColNames = ['Label']+['I{}'.format(i) for i in range(0,2)]+['C{}'.format(i) for i in range(0,2)]
    for oldName, newName in zip(oldColNames, newColNames):
        df = df.withColumnRenamed(oldName, newName)
    return df


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
    grad = augmentedData.map(lambda x: (W.dot(x[0]) - x[1])*x[0]).mean() * 2
    new_model = W - learningRate * grad
    ################## (END) YOUR CODE ################# 
    
    return new_model

w = np.array([8, -3, -1, 3, 8])
df = generateToyDataset(w)


def dfToRDD(row):
    '''
    Converts dataframe row to rdd format.
        From: DataFrame['Label', 'I0', ..., 'C0', ...]
        To:   (features_array, y)
    '''
    features_array = [row['I{}'.format(i)] for i in range(0, 2)] + [row['I{}'.format(i)] for i in range(0, 2)]
    y = row['Label']
    return (features_array, y)
    
trainRDD = df.rdd.map(dfToRDD)

print(trainRDD.collect())