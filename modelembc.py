from __future__ import division
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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
spark = SparkSession \
    .builder \
    .appName("Python Spark training model") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

path_trainingset = 'parquet/dataset100.parquet'
path_testset = 'parquet/testset10.parquet'
#Path per lettura da hdfs
training_data = spark.read.parquet('hdfs:/bigdata/' + path_trainingset)
test_data = spark.read.parquet('hdfs:/bigdata/' + path_trainingset)

#path per la lettura da locale
#training_data = spark.read.parquet('/homo/ubuntu/data' + path_trainingset)
#test_data = spark.read.parquet('/homo/ubuntu/data' + path_trainingset)

#Classificatore
lr = LogisticRegression(regParam = 0.02, maxIter = 20)

#Pipeline, in questo caso comprende solo il classificatore
steps = [lr]
pipeline = Pipeline(stages=steps)

#Istanzio i parametri per il crossvalidation
paramGrid = ParamGridBuilder().build()

#Classificatore di tipo binario
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")

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
#cvModel = pipeline.fit(training_data)
#predictions = cvModel.transform(test_data)
print ('Training finito con un tempo in minuti di: ', (time.time() - t0) / 60)
areaUnderROC = evaluator.evaluate(predictions)

#Calcolo delle metriche per la valutazione del modello
lp = predictions.select("label", "prediction")
counttotal = predictions.count()
correct = lp.filter(lp.label == lp.prediction).count()
wrong = lp.filter(lp.label != lp.prediction).count()
ratioWrong = wrong / counttotal
lp = predictions.select( "prediction","label")
counttotal = predictions.count()
correct = lp.filter(lp.label == lp.prediction).count()
wrong = lp.filter("label != prediction").count()
ratioWrong = wrong / counttotal
ratioCorrect = correct / counttotal
truen =( lp.filter(lp["label"] == 0.0).filter(lp.label == lp.prediction).count()) / counttotal
truep = (lp.filter(lp["label"] == 1.0).filter(lp.label == lp.prediction).count()) / counttotal
falsen = (lp.filter(lp["label"] == 0.0).filter(lp.label != lp.prediction).count()) / counttotal
falsep = (lp.filter(lp["label"] == 1.0).filter(lp.label != lp.prediction).count()) / counttotal

precision= truep / (truep + falsep)
recall= truep / (truep + falsen)
fmeasure= 2 * ((precision * recall )/ (precision + recall))
accuracy=(truep + truen) / (truep + truen + falsep + falsen)

#Salvo su file tutte le metriche calcolate
with open('results.txt',"a") as f:
    f.write("Training con embedding pre calcolati (csv), con un training set di  " + str(training_data.count()) + ' elementi \n')
    f.write("counttotal: " + str(counttotal) + "\n")
    f.write("correct: " + str(correct) + "\n")
    f.write("wrong: " + str(wrong) + "\n")
    f.write("ratioWrong: " + str(ratioWrong) + "\n")
    f.write("ratioCorrect: " + str(ratioCorrect) + "\n")
    f.write("truen: " + str(truen) + "\n")
    f.write("truep: " + str(truep) + "\n")
    f.write("falsen: " + str(falsen) + "\n")
    f.write("falsep: " + str(falsep) + "\n")
    f.write("precision: " + str(precision) + "\n")
    f.write("recall: " + str(recall) + "\n")
    f.write("fmeasure: " + str(fmeasure) + "\n")
    f.write("accuracy: " + str(accuracy) + "\n")
    f.write("areaUnderROC: " + str(areaUnderROC) + "\n")
    f.write('Training finito con un tempo in minuti di: ' + str((time.time() - t0) / 60) + "\n")
