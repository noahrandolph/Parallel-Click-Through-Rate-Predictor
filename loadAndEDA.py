#!/usr/bin/env python

from pyspark.sql import types
from pyspark.sql.functions import udf, col, countDistinct, isnan, when, count, desc
import pandas as pd
from pyspark.mllib.stat import Statistics

MAINCLOUDPATH = 'gs://w261_final_project/train.txt'
MINICLOUDPATH = 'gs://w261_final_project/train_005.txt'
MINILOCALPATH = 'data/train_005.txt'

SEED = 2615


# start Spark Session
from pyspark.sql import SparkSession
app_name = "loadAndEDA"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
sc = spark.sparkContext


def loadData():
    '''load the data into a Spark dataframe'''
    # select path to data: MAINCLOUDPATH; MINICLOUDPATH; MINILOCALPATH
    df = spark.read.csv(path=MINILOCALPATH, sep='\t')
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
    '''returns head of the training dataset'''
    return df.head(n)


def getMedians(df, cols):
    '''returns approximate median values of the columns given, with null values ignored'''
    # 0.5 relative quantile probability and 0.05 relative precision error
    return df.approxQuantile(cols, [0.5], 0.05)

def getDescribe(df, cols):
    return df.select(cols).describe().show()

def getDistinctCount(df, cols):
    return df.agg(*(countDistinct(col(c)).alias(c) for c in cols)).show()

def checkNA(df, cols):
    return df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in cols]).show()

def getCorrMatrix(df, cols):
    df = df.select(cols)
    col_names = df.columns
    features = df.rdd.map(lambda row: row[0:])
    corr_mat=Statistics.corr(features, method="pearson")
    corr_df = pd.DataFrame(corr_mat)
    corr_df.index, corr_df.columns = col_names, col_names
    return corr_df
    
    
df = loadData().cache()
testDf, trainDf = splitIntoTestAndTrain(df)
#print("\nTEST DATASET ROW COUNTS: ", testDf.count())
#print("\nTRAIN DATASET ROW COUNTS: ", trainDf.count())
## print("HEAD\n", displayHead(trainDf))
#print("\nCOLUMN TYPES\n", df.dtypes)
#print("\nMEDIAN OF NUMERIC COLUMNS\n", getMedians(trainDf, trainDf.columns[1:14]))

#print("\nDESCRIPTIONS OF NUMERICAL COLUMNS 1-7\n")
#getDescribe(trainDf, trainDf.columns[1:8])
#print("\nDESCRIPTIONS OF NUMERICAL COLUMNS 8-14\n")
#getDescribe(trainDf, trainDf.columns[8:15])
#print("\nCOUNTS OF DISTINCT VALUE FOR CATEGORICAL VARIABLE COLUMNS")
#getDistinctCount(trainDf, trainDf.columns[15:])

#print("\nCOUNTS OF NAs FOR COLUMN 0 - 19")
#checkNA(trainDf, trainDf.columns[:20])
#print("\nCOUNTS OF NAs FOR COLUMN 20 - 39")
#checkNA(trainDf, trainDf.columns[20:])
getCorrMatrix(trainDf, trainDf.columns[1:14])