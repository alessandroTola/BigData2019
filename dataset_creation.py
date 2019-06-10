import time
from pyspark import SparkContext
from pyspark.sql import SQLContext
from nltk.corpus import stopwords
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.feature import HashingTF, Tokenizer, IDF
from pyspark.ml.feature import StopWordsRemover
from gensim.models import KeyedVectors
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
import pandas as pd
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import *
import DBUtils

def g(x):
    print x

#Creazione del contex di spark
sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("local[8]").getOrCreate()

#variabile grobale contentente gli embeddings per calcolati
wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv('sentic2vec.csv', skiprows=1, encoding='latin').iterrows()})

#Funzione che richerca e calcola gli embaddings per ogni recensione
def sum_vectors(words):
    features_accumulator = []
    for word in words:
        try:
            features_accumulator.append(wordDict.value[word])
        except:
            pass
    if(len(features_accumulator) == 0):
        return None
    return Vectors.dense(np.sum(features_accumulator, axis=0, dtype=np.float64))
t0 = time.time()

#Path file json
datapath = 'training_set_BIG'
test = 'test_set_BIG'

#Creo dataFrame da json con i campi text label
reviews = sqlContext.read.json(datapath)
reviews_test = sqlContext.read.json(test)

#tokenize the text
regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'reviewTS', outputCol = 'reviewTokensUf')
reviews_token = regexTokenizer.transform(reviews).drop('reviewTS')
reviews_token_test = regexTokenizer.transform(reviews_test).drop('reviewTS')

# remove stopwords
swr = StopWordsRemover(inputCol = 'reviewTokensUf', outputCol = 'reviewTokens')
reviews_swr = swr.transform(reviews_token).drop('reviewTokensUf')
reviews_swr_test = swr.transform(reviews_token_test).drop('reviewTokensUf')

# Calculate the feature using word embedding pre calc, drop null record
sum_vectors_udf = udf(sum_vectors, VectorUDT())

#Calcolo le features, creo il detaset pronto per l'addestramento
vec_df = reviews_swr.withColumn('features', sum_vectors_udf('reviewTokens')).drop('reviewTokens').dropna(subset=['features'])
vec_df_test = reviews_swr_test.withColumn('features', sum_vectors_udf('reviewTokens')).drop('reviewTokens').dropna(subset=['features'])

print ('Fine creazione vettore: ', (time.time() - t0) / 60)

#Salvo il dataset nel formato parquet, più veloce in lettura
vec_df.write.mode('overwrite').parquet("parquet/datasetG.parquet")
vec_df_test.write.mode('overwrite').parquet("parquet/testsetG.parquet")
