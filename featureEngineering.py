#!/usr/bin/env python

from pyspark.sql import types
from pyspark.sql.functions import udf
from pyspark.sql.functions import desc
import numpy as np


MAINCLOUDPATH = 'gs://w261_final_project/train.txt'
TOYCLOUDPATH = 'gs://w261_final_project/train_005.txt'
TOYLOCALPATH = 'data/train_005.txt'
NUMERICCOLS = 13
CATEGORICALCOLS = 26
SEED = 2615


# start Spark Session
from pyspark.sql import SparkSession
app_name = "logisticRegression"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext


def loadData():
    '''load the data into a Spark dataframe'''
    # select path to data: MAINCLOUDPATH; TOYCLOUDPATH; TOYLOCALPATH
    df = spark.read.csv(path=TOYLOCALPATH, sep='\t')
    # change column names
    oldColNames = df.columns
    newColNames = ['Label']+['I{}'.format(i) for i in range(0,NUMERICCOLS)]+['C{}'.format(i) for i in range(0,CATEGORICALCOLS)]
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


def getMedians(df, cols):
    '''
    returns approximate median values of the columns given, with null values ignored
    '''
    # 0.5 relative quantile probability and 0.05 relative precision error
    return df.approxQuantile(cols, [0.5], 0.05)


def getMostFrequentCats(df, cols, n):
    '''
    returns a dict where the key is the column and value is an ordered list
    of the top n categories in that column in descending order
    '''
    freqCatDict = {col: None for col in df.columns[cols:]}
    for col in df.columns[cols:]:
        listOfRows = df.groupBy(col).count().sort('count', ascending=False).take(n)
        topCats = [row[col] for row in listOfRows]
        freqCatDict[col] = topCats[:n]
    return freqCatDict
    
    
def dfToRDD(row):
    '''
    Converts dataframe row to rdd format.
        From: DataFrame['Label', 'I0', ..., 'C0', ...]
        To:   (features_array, y)
    '''    
    features_list = [row['I{}'.format(i)] for i in range(0, NUMERICCOLS)] + [row['C{}'.format(i)] for i in range(0, CATEGORICALCOLS)]
    features_array = np.array(features_list)
    y = row['Label']
    return (features_array, y)



df = loadData()
testDf, trainDf = splitIntoTestAndTrain(df)
testDf.cache()
trainDf.cache()

# get dict of median numeric values for imputation of missing values
fillNADictNum = {key: value for (key, value) in zip(trainDf.columns[1:NUMERICCOLS+1], 
                                                    [x[0] for x in getMedians(trainDf, trainDf.columns[1:NUMERICCOLS+1])])}

# get top n most frequent categories for each column
mostFreqCatDict = getMostFrequentCats(trainDf, NUMERICCOLS+1, 10)


trainRDD = trainDf.rdd.map(dfToRDD).cache()

print(mostFreqCatDict)