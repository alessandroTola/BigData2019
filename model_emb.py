from __future__ import division
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
import nltk
import time
from pyspark.sql.functions import when
import pandas as pd
import numpy as np
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
import sys
import os

#Funzione per la ricerca degli embedding pre calcolati e calcolo delle features
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

sc = SparkContext()
sqlContext = SQLContext(sc)
nltk.download("stopwords")
stopwordList = nltk.corpus.stopwords.words('english')

spark = SparkSession \
    .builder \
    .appName("Python Spark create dataset") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#Creo una variabile globali di broadcast per il calcolo delle features da csv
wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv('sentic2vec.csv', skiprows=1, encoding='latin').iterrows()})
#Creo una variabile globali di broadcast per il calcolo delle features da txt
#wordDict = spark.sparkContext.broadcast({row[0] : np.asarray(row[1:]) for _, row in pd.read_csv('embeddings_snap_s128_e15.txt', skiprows=1, sep=" ", header=None).iterrows()})

#training_set_file = 'training_set_multiclass'
#test_set_file = 'test_set_multiclass'

training_set_file = 'training_set'
test_set_file = 'test_set'

training_data = spark.read.format("json").load(training_set_file)
test_data = spark.read.format("json").load(test_set_file)

print 'Dataset dimendion ' + str(training_data.count())
print 'Testset dimendion ' + str(test_data.count())

#Applico tokenixer al testo per splittarlo il parole
regexTokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'reviewTS', outputCol = 'reviewTokensUf')
training_data = regexTokenizer.transform(training_data).drop('reviewTS')
test_data = regexTokenizer.transform(test_data).drop('reviewTS')

#Rimuovo le parole poco utili come congiunzioni
swr = StopWordsRemover(inputCol = 'reviewTokensUf', outputCol = 'reviewTokens', stopWords = stopwordList)
training_data = swr.transform(training_data).drop('reviewTokensUf')
test_data = swr.transform(test_data).drop('reviewTokensUf')

#Calcolo delle features
sum_vectors_udf = udf(sum_vectors, VectorUDT())
training_data = training_data.withColumn('features', sum_vectors_udf('reviewTokens')).\
drop('reviewTokens').dropna(subset=['features'])
test_data = test_data.withColumn('features', sum_vectors_udf('reviewTokens')).\
drop('reviewTokens').dropna(subset=['features'])
training_data.show(10)

time_embeddings = (time.time() - t0) / 60)
#training_data.write.mode('overwrite').json('training_set_emblll')
#test_data.write.mode('overwrite').json('test_set_emblll')
#training_data.write.mode('overwrite').parquet("parquet/datasetcsv_multiclass.parquet")
#test_data.write.mode('overwrite').parquet("parquet/testsetcsv_multiclass.parquet")

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
time_training = (time.time() - t0) / 60)

#Salvataggio modello
cvModel.bestModel.write().overwrite().save('models/logisticregression_best_model_EMB')
#Per caricare il modello salvato
#cvModel = PipelineModel.load('models/logisticregression_best_model_EMB')

# Make predictions on test documents. cvModel uses the best model found (lrModel).
predictions = cvModel.transform(test_data)
time_prediction = (time.time() - t0) / 60)

#LOGICREGRETION
#model = pipeline.fit(training_data)
#predictions = model.transform(test_data)
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
    f.write("time embeddings: " + str(time_embeddings) + "\n")
    f.write("time training: " + str(time_training) + "\n")
    f.write("time prediction: " + str(time_prediction) + "\n")
