#!/usr/bin/env python

MAINPATH = 'gs://w261_final_project/train.txt'
TOYPATH = 'gs://w261_final_project/train_005.txt'
SEED = 2615

dataPath = TOYPATH

# start Spark Session
from pyspark.sql import SparkSession
app_name = "row_counts"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
        
sc = spark.sparkContext


def splitIntoTestAndTrain():
    '''randomly splits 80/20 into training and testing dataframes'''
    # load the data into Spark dataframes
    df = spark.read.csv(dataPath, sep='\t')
    # split into test and training data
    splits = df.randomSplit([0.2, 0.8], seed=SEED)
    testDf = splits[0]
    trainDf = splits[1]
    return testDf, trainDf


def displayHead(df, n=5):
    '''prints out head of the training dataset'''
    return df.head(n)


testDf, trainDf = splitIntoTestAndTrain()


print("TEST DATASET ROW COUNTS: ", testDf.count())
print("TRAIN DATASET ROW COUNTS: ", trainDf.count())
print("HEAD\n", displayHead(trainDf))