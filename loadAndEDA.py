#!/usr/bin/env python

from pyspark.sql import types
from pyspark.sql.functions import udf

MAINCLOUDPATH = 'gs://w261_final_project/train.txt'
TOYCLOUDPATH = 'gs://w261_final_project/train_005.txt'
TOYLOCALPATH = 'data/train_005.txt'
SEED = 2615


# start Spark Session
from pyspark.sql import SparkSession
app_name = "row_counts"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext


def loadData():
    # load the data into Spark dataframes
    # select path to data: MAINCLOUDPATH; TOYCLOUDPATH; TOYLOCALPATH
    df = spark.read.csv(path=TOYLOCALPATH, sep='\t')
    # change column names
    oldColNames = df.columns
    newColNames = ['Label']+['I{}'.format(i) for i in range(0,13)]+['C{}'.format(i) for i in range(0,26)]
    for oldName, newName in zip(oldColNames, newColNames):
        df = df.withColumnRenamed(oldName, newName)
    # change int column types to int from string
    for col in df.columns[:14]:
        df = df.withColumn(col, df[col].cast('int'))
    return df


def splitIntoTestAndTrain(df):
    '''randomly splits 80/20 into training and testing dataframes'''
    splits = df.randomSplit([0.2, 0.8], seed=SEED)
    testDf = splits[0]
    trainDf = splits[1]
    return testDf, trainDf


def displayHead(df, n=5):
    '''prints out head of the training dataset'''
    return df.head(n)


df = loadData()
testDf, trainDf = splitIntoTestAndTrain(df)
print("TEST DATASET ROW COUNTS: ", testDf.count())
print("TRAIN DATASET ROW COUNTS: ", trainDf.count())
print("HEAD\n", displayHead(trainDf))
print(df.dtypes)