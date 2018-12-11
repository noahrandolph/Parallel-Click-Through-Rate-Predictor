import numpy as np
from sklearn import datasets



class OneHotEncoder:

    """
    Class to assist one hot encoding a category based column using numpy
    
    Attributes:
        
        dense:  A dense representation of the one hot encoded column
                 in the form (N,[C],[V])  
                Where:  
                N is the number of columns in the representation
                C is an array of columns that are activated
                V is the value for the activated column.
                e.g. (10,[5][1]) represents a single row of one hot encoded
                values where there are 10 columns total, and column 5 has a value of 1
                
        sparse:  A sparse representation of the category column.  Essentially
                A matrix of 0/1 values.  The matrix has the same number of columns
                as the number of unique categories, and the number of rows 
                as the category column
                
        indexes:  A list of numeric indexes corresponding to unique categories.
                 e.g. category column ['a','b','c','a','a'] 
                 index column [1,2,3,1,1]
    """

    # initialize properties.  Set up the lookup of unique values to categories
    def __init__(self, catagories):
        self.cats = catagories
        self.lookup = {k: float(v) for k, v in zip(np.unique(self.cats), range(0, len(np.unique(self.cats))))}
        self.__set_sparse(None)
        self.__set_dense(None)
        self.__set_indexes(None)

    # private. set indexes property
    def __set_indexes(self,val):
        self.__indexes = val

    # lazy create index list
    def get_indexes(self):
        if self.__indexes is None:
            self.__indexes = [self.lookup[x[0]] for x in self.cats]
        return self.__indexes

    # private.  set sparse property
    def __set_sparse(self,val):
        self.__sparse = val

    # return a sparse representation of the category column.
    # eg  category column ['a','b','c','a','a']
    # return [[1,0,0],
    #         [0,1,0],
    #         [0,0,1],
    #         [1,0,0],
    #         [1,0,0]]
    def get_sparse(self):
        if self.__sparse is None:
            self.__sparse = np.zeros((X.shape[0], len(np.unique(self.indexes ))))
            for i,x in enumerate(self.indexes):
                self.__sparse[i][int(x)] = 1.0
        return(self.__sparse)

    # private.  Sets a dense reprsentation of category column
    def __set_dense(self, val):
        self.__dense = val

    # Returns a dense representation of category column.
    #eg category column['a', 'b', 'c', 'a', 'a']
    # return [(3,[0],[1]),
    #         (3,[1],[1]),
    #         (3,[2],[1]),
    #         (3,[0],[1]),
    #         (3,[0],[1])]
    def get_dense(self):
        if self.dense is None:
            self.dense = [(len(np.unique(self.indexes)),[self.lookup[x[0]]],[1.0]) for x in self.cats]
        return(self.dense)

    indexes = property(get_indexes, __set_indexes)
    sparse = property(get_sparse, __set_sparse)
    dense = property(get_dense, __set_dense)

iris = datasets.load_iris()
X = iris.data

cats = np.random.choice((['a','b','c','d','e']),X.shape[0] ).reshape((X.shape[0],1))
ohe = OneHotEncoder(cats)
X = np.concatenate((X,cats),axis=1)
X = np.concatenate((X,ohe.get_sparse()),axis=1)

print(X)
print(X.shape)
print(X[0:][6:10])

