from __future__ import division
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, HashingTF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import nltk
import time
from pyspark.sql.functions import when
import sys
import os

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

training_set_file = 'training_setlll'
test_set_file = 'test_setlll'

training_data = spark.read.format("json").option("inferSchema", "true").load(training_set_file)
test_data = spark.read.format("json").option("inferSchema", "true").load(test_set_file)

print 'Dataset dimendion ' + str(training_data.count())
print 'Testset dimendion ' + str(test_data.count())

#Caclolo delle features Inverse document frequency (IDF) dopo averle ragruppate e contate con CountVectorizer
tokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'reviewTS', outputCol = 'reviewTokensUf')
remover = StopWordsRemover(inputCol = 'reviewTokensUf', outputCol = 'reviewTokens', stopWords = stopwordList)
count_vector = CountVectorizer(inputCol = 'reviewTokens', outputCol = 'cv', vocabSize = 200000, minDF = 2.0)
hashingTF = HashingTF(inputCol="cv", outputCol="htf", numFeatures=100)
idf = IDF(inputCol="htf", outputCol="features")
lr = LogisticRegression(regParam = 0.02, maxIter = 20)

steps = [tokenizer, remover, count_vector, idf,lr]
pipeline = Pipeline(stages=steps)

#Crossval parameter
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
cvModel.bestModel.write().overwrite().save('models/logisticregression_best_model_IDF')

#Per caricare il modello salvato
#cvModel = PipelineModel.load('models/logisticregression_best_model_IDF')

# Make predictions on test documents. cvModel uses the best model found (lrModel).
predictions = cvModel.transform(test_data)
time_prediction = (time.time() - t0) / 60)

#LOGICREGRETION
#model = pipeline.fit(training_data)
#predictions = model.transform(test_data)

#Calcolo delle metriche per la valutazione del modello
areaUnderROC = evaluator.evaluate(predictions)

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
    f.write("Training calcolando le features, con un training set di  " + str(training_data.count()) + ' elementi \n')
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
    f.write("time training: " + str(time_training) + "\n")
    f.write("time prediction: " + str(time_prediction) + "\n")
