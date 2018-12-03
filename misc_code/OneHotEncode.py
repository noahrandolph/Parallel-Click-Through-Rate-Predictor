from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer

# start Spark Session
app_name = "FinalProject_OneHotEncode"
master = "local[*]"
spark = SparkSession\
        .builder\
        .appName(app_name)\
        .master(master)\
        .getOrCreate()
sc = spark.sparkContext

# Run locally
df = spark.read.csv('toy_100.tsv',sep='\t')

# Run in cloud
# df = spark.read.csv('gs://tabbone/train.txt',sep='\t')

# rename columns from default _c0,_c1... _c26 to ['Label','I0','I1'...'C0','C1'...]
# per the readme.
#

# Get default column names
old_cols=df.columns

# List of integer column names ['I0','I1','I2'...]
num_cols = ['I{}'.format(i) for i in range(0,13)]

# List of categorical column names ['C1','C2','C3'...]
cat_cols = ['C{}'.format(i) for i in range(0,26)]

# Single list of all column names
new_cols = ['Label'] + num_cols + cat_cols

# rename columns
for old_name,new_name in zip(old_cols,new_cols):
    df = df.withColumnRenamed(old_name,new_name)

# Categories must be converted into indexes before being one hot encoded.
# Create a list of index column names to append to the DF, 1 for each category
# in form ['C0_IDX','C1_IDX'...]
cat_idx_cols = ['{}_IDX'.format(colName) for colName in cat_cols]

# Column names for the one hot encoded indexes in form ['C0_OHE','C1_OHE','C2_OHE'].
ohe_cat_cols = ['{}_OHE'.format(colName) for colName in cat_cols]

# Create StringIndexer transformer.  The transformer will look through each value in
# input column (a string) and create a unique double index.  Null values are assumed
# to be a single category and are not discarded
#
# Loop through each category column and populate a corresponding category index column.
#
# +--------+--------+--------+	    +------+------+------+
# |      C0|      C1|      C2| .... |C0_IDX|C1_IDX|C2_IDX|
# +--------+--------+--------+	    +-----+----+---+----+-
# |05db9164|403ea497|2cbec47f| .... |   0.0|  34.0|  43.0|
# |05db9164|421b43cd|e8c51b7b| .... |   0.0|  32.0|  44.0|
# |be589b51|0a519c5c|ad4b77ff| .... |   3.0|   4.0|   1.0|

for col,index_col in zip(cat_cols,cat_idx_cols):
    catIndexer = StringIndexer(inputCol = col,outputCol=index_col,handleInvalid = 'keep')
    df = catIndexer.fit(df).transform(df)


# Create a OneHotEncoderEstimator.  The 'Estimator' part just means it
# does an initial pass through before calculations.  It does not actually make
# an estimate.
# Provided a list of index columns, it will produce corresponding output columns
# in dense format.
#
#
# +--------+--------+--------+	    +------+------+------+ 	    +---------------+-------------+----------------+
# |      C0|      C1|      C2| .... |C0_IDX|C1_IDX|C2_IDX| .... |         C0_OHE|       C1_OHE|          C2_OHE|
# +--------+--------+--------+	    +-----+----+---+----+-      +---------------+-------------+---------------+
# |05db9164|403ea497|2cbec47f| .... |   0.0|  34.0|  43.0| .... | (19,[0],[1.0])|(54,[34],[1.0]) |(90,[43],[1.0])
# |05db9164|421b43cd|e8c51b7b| .... |   0.0|  32.0|  44.0| .... | (19,[0],[1.0])|(54,[32],[1.0]) |(90,[44],[1.0])
# |be589b51|0a519c5c|ad4b77ff| .... |   3.0|   4.0|   1.0| .... | (19,[3],[1.0])|(54,[4],[1.0])  |(90,[1],[1.0])
#
# (19,0,[1.0]) = 19 columns.  Row 0 has a value of 1.0
#


ohee = OneHotEncoderEstimator(
    inputCols = cat_idx_cols,
    outputCols = ohe_cat_cols
)

ohee.fit(df).transform(df).show()

