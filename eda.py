#!/usr/bin/env python

# start Spark Session
from pyspark.sql import SparkSession
app_name = "eda"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .getOrCreate()
        
sc = spark.sparkContext

# load the data into Spark dataframes
df = spark.read.csv('gs://w261_final_project/train.txt', sep='\t')

# split into test and training data
splits = df.randomSplit([0.2, 0.8], seed=2615)
testDf = splits[0]
trainDf = splits[1]

print(df.columns)
print(testDf.count())
print(trainDf.count())