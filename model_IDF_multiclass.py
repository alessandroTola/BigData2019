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

training_set_file = 'training_set_multiclass'
test_set_file = 'test_set_multiclass'

training_data = spark.read.format("json").option("inferSchema", "true").load(training_set_file)
test_data = spark.read.format("json").option("inferSchema", "true").load(test_set_file)

print 'Dataset dimendion ' + str(training_data.count())
print 'Testset dimendion ' + str(test_data.count())

#Caclolo delle features Inverse document frequency (IDF) dopo averle ragruppate e contate con CountVectorizer
tokenizer = RegexTokenizer(gaps = False, pattern = '\w+', inputCol = 'reviewTS', outputCol = 'reviewTokensUf')
remover = StopWordsRemover(inputCol = 'reviewTokensUf', outputCol = 'reviewTokens', stopWords = stopwordList)
count_vector = CountVectorizer(inputCol = 'reviewTokens', outputCol = 'cv', vocabSize = 200000, minDF = 2.0)
idf = IDF(inputCol="cv", outputCol="features")
#lr = LogisticRegression(regParam = 0.02, maxIter = 20)
#lr = RandomForestClassifier(maxDepth=5, numTrees=20, seed=42)
#lr = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=30)
lr = DecisionTreeClassifier(labelCol="label", featuresCol="features")
steps = [tokenizer, remover, count_vector, idf,lr]
pipeline = Pipeline(stages=steps)

#Crossval parameter
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

#Calcolo delle metriche per la valutazione del modello
print ('Training finito con un tempo in minuti di: ', (time.time() - t0) / 60)
print ('Accuracy multiclass: ', evaluator.evaluate(predictions))
