from __future__ import division
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import nltk
import time
from pyspark.sql.functions import when
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT


t0 = time.time()

sc = SparkContext()
sqlContext = SQLContext(sc)
spark = SparkSession.builder.master("local[8]").getOrCreate()

training_data = spark.read.format("json").load('training_set_emb_multiclass')
test_data = spark.read.format("json").load('test_set_emb_multiclass')
#training_data = spark.read.parquet("parquet/datasetcsv_multiclass.parquet")
#test_data = spark.read.parquet("parquet/testsetcsv_multiclass.parquet")

#Classificatore
#lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
#lr = RandomForestClassifier(maxDepth=8, numTrees=40, seed=42)
lr = DecisionTreeClassifier(labelCol="label", featuresCol="features")
#Pipeline, in questo caso comprende solo il classificatore
steps = [lr]
pipeline = Pipeline(stages=steps)

#Istanzio i parametri per il crossvalidation
paramGrid = ParamGridBuilder().build()

#Classificatore di tipo binario
evaluator = MulticlassClassificationEvaluator(metricName='accuracy')

#Cross validation con 10 fold
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=10,
                          seed=42)
#CROSSVALIDATION
cvModel = crossval.fit(training_data)

# Make predictions on test documents. cvModel uses the best model found (lrModel).
predictions = cvModel.transform(test_data)

#LOGICREGRETION
#model = pipeline.fit(training_data)
#predictions = model.transform(test_data)
print ('Training finito con un tempo in minuti di: ', (time.time() - t0) / 60)
print ('Accuracy multiclass: ', evaluator.evaluate(predictions))
