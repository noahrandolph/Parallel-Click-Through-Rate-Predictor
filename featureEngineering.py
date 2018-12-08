#!/usr/bin/env python

from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, desc, isnan, when
import numpy as np
from operator import add
import copy


MAINCLOUDPATH = 'gs://w261_final_project/train.txt'
TOYCLOUDPATH = 'gs://w261_final_project/train_005.txt'
TOYLOCALPATH = 'data/train_005.txt'
NUMERICCOLS = 13
CATEGORICALCOLS = 26
NUMERICCOLNAMES = ['I{}'.format(i) for i in range(0,NUMERICCOLS)]
CATCOLNAMES = ['C{}'.format(i) for i in range(0,CATEGORICALCOLS)]
SEED = 2615


# start Spark Session
from pyspark.sql import SparkSession
app_name = "featureEngineering"
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
    newColNames = ['Label'] + NUMERICCOLNAMES + CATCOLNAMES
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
    

def rareReplacer(df, dictOfMostFreqSets):
    '''
    Iterates through columns and replaces non-Frequent categories with 'rare' string.
    '''
    for colName in df.columns[NUMERICCOLS+1:]:
        bagOfCats = dictOfMostFreqSets[colName]
        df = df.withColumn(colName, udf(lambda x: 'rare' if x not in bagOfCats else x, StringType())(df[colName])).cache()
    return df

    
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


def emitColumnAndCat(line):
    """
    Takes in a row from RDD and emits a record for each categorical column value along with a zero for one-hot encoding.
    The emitted values will become a reference dictionary for one-hot encoding in later steps.
        Input: (array([features], dtype='<U21'), 0) or (features, label)
        Output: ((categorical column, category), 0) or (complex key, value)
    The last zero in the output is for initializing one-hot encoding.
    """
    elements = line[0][NUMERICCOLS:]
    for catColName, element in zip(CATCOLNAMES, elements):
        yield ((catColName, element), 0)


def oneHotEncoder(line):
    """
    Takes in a row from RDD and emits row where categorical columns are replaced with 1-hot encoded columns.
        Input: (numerical and categorical features, label)
        Output: (numerical and one-hot encoded categorical features, label)
    """
    oneHotDict = copy.deepcopy(oneHotReference)
    elements = line[0][NUMERICCOLS:]
    for catColName, element in zip(CATCOLNAMES, elements):
        oneHotDict[(catColName, element)] = 1
    numericElements = list(line[0][:NUMERICCOLS])
    return (numericElements + [value for key, value in oneHotDict.items()], line[1])


# load data
df = loadData()
testDf, trainDf = splitIntoTestAndTrain(df)
testDf.cache()
trainDf.cache()

# get top n most frequent categories for each column (in training set only)
n = 10
mostFreqCatDict = getMostFrequentCats(trainDf, NUMERICCOLS+1, n)

# get dict of sets of most frequent categories in each column for fast lookups during filtering (in later code)
setsMostFreqCatDict = {key: set(value) for key, value in mostFreqCatDict.items()}

# get the top category from each column for imputation of missing values (in training set only)
fillNADictCat = {key: (value[0] if value[0] is not None else value[1]) for key, value in mostFreqCatDict.items()}

# get dict of median numeric values for imputation of missing values (in training set only)
fillNADictNum = {key: value for (key, value) in zip(trainDf.columns[1:NUMERICCOLS+1], 
                                                    [x[0] for x in getMedians(trainDf,
                                                                              trainDf.columns[1:NUMERICCOLS+1])])}

# impute missing values in training and test set
trainDf = trainDf.na.fill(fillNADictNum) \
                 .na.fill(fillNADictCat).cache()
testDf = testDf.na.fill(fillNADictNum) \
               .na.fill(fillNADictCat).cache()

# replace low-frequency categories with 'rare' string in training and test set
trainDf = rareReplacer(trainDf, setsMostFreqCatDict) # df gets cached in function
testDf = rareReplacer(testDf, setsMostFreqCatDict) # df gets cached in function

# # numerically index categorical columns for one-hot encoder and to combine rare categories into one
# for catColumn in trainDf.columns[NUMERICCOLS+1:]:
#     catIndexer = StringIndexer(inputCol=catColumn, outputCol=catColumn+'Index', handleInvalid='error') # forces you to have different in & out col names
#     stringIndexerModel = catIndexer.fit(trainDf)
#     trainDf = stringIndexerModel.transform(trainDf).drop(catColumn).cache() # original string columns are kept in dataframe so should be deleted
#     testDf = stringIndexerModel.transform(testDf).drop(catColumn).cache()
    
# # convert categorical columns to 1 hot encoded columns
# indexColumnNames = trainDf.columns[NUMERICCOLS+1:]
# oneHotColumnNames = [column.replace('Index', '') for column in trainDf.columns[NUMERICCOLS+1:]]
# oneHotEncoder = OneHotEncoderEstimator(inputCols=indexColumnNames, outputCols=oneHotColumnNames) # forces you to have different in & out column names
# oneHotModel = oneHotEncoder.fit(trainDf)                                                                        
# trainDf = oneHotModel.transform(trainDf).cache()
# testDf = oneHotModel.transform(testDf).cache()

# # drop the index columns (original string columns are kept in dataframe so should be deleted)
# for column in indexColumnNames:
#     trainDf = trainDf.drop(column) 
#     testDf = testDf.drop(column)

# # convert SparseVectors to 1D arrays in order to convert dataframe to RDD    
# for column in trainDf.columns[NUMERICCOLS+1:]:
#     trainDf = trainDf.withColumn(column, udf(lambda x: list(OrderedDict((y, None) for y in x)))(trainDf[column])).cache()

# convert dataframe to RDD 
trainRDD = trainDf.rdd.map(dfToRDD).cache()
testRDD = testDf.rdd.map(dfToRDD).cache()
        
# create and broadcast reference dictionary to be used in constructing 1 hot encoded RDD
oneHotReference = trainRDD.flatMap(emitColumnAndCat) \
                          .reduceByKeyLocally(add) # note: only the zero values are being added here (main goal is to output a dictionary)
sc.broadcast(oneHotReference)

# replace rows with new rows having categorical columns 1-hot encoded
trainRDD1Hot = trainRDD.map(oneHotEncoder)


print(trainRDD1Hot.takeSample(False, 5, SEED))