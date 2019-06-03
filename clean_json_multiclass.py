from __future__ import division
from pyspark import SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.feature import RegexTokenizer, Bucketizer
from pyspark.sql.functions import concat, col, lit


sc = SparkContext()
sqlContext = SQLContext(sc)

spark = SparkSession \
    .builder \
    .appName("Python Spark create dataset") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

datapath = 'datasetGRANDE.json'

dataset = spark.read.format("json").option("inferSchema", "true").load(datapath)

#Creo una nuova colonna contenente la concatenazioni tra titolo della recensione e testo della recensione
dataset = dataset.withColumn("reviewTS",
  concat(dataset["summary"], lit(" "),dataset["reviewText"])).drop("helpful").drop("reviewerID").\
  drop("reviewerName").drop("reviewTime").drop("asin").drop("unixReviewTime")

#dataset.show(10)

#Creo un Bucketizer per la suddivisione delle classi, da -1.0 a 4.0 (recensioni con 1.0, 2.0)
# da 4.0 a 6.0 (recensioni con 4.0, 5.0)
splits = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
bucketizer = Bucketizer(splits=splits, inputCol="overall", outputCol="label")
dataset = bucketizer.transform(dataset)

#Ripulisco il dataset droppando le colonne che non mi servono, tengo solo label e testo
dataset.groupBy("overall","label").count().show()
dataset = dataset.drop('overall').drop('reviewText').drop('summary')

#Calcolo rapporto tra classe positiva e negativa da utilizzare per l'undersampling
#ratio = dataset.filter(dataset["label"] == 0.0).count() / dataset.filter(dataset["label"] == 1.0).count()

#Applico l'undersampler utilizzando il rapporto tra le classi calcolato in precedenza
dataset = dataset.sampleBy("label", fractions={0.0: 1.0, 1.0: 1.0, 2.0: 0.5, 3.0: 0.2, 4.0: 0.1}, seed=42)
dataset.groupBy("label").count().show()

#Splitto il dataset in training_set e test_set tenendo 80% per training_set e 20% test_set
[trainingData, testData] = dataset.randomSplit([0.5, 0.1], 42)

print 'Dataset dimendion ' + str(trainingData.count())
print 'Testset dimendion ' + str(testData.count())
dataset.show(10)

#Salvo i nuovi json
trainingData.write.mode('overwrite').json('training_set_multiclass')
testData.write.mode('overwrite').json('test_set_multiclass')

trainingData.groupBy("label").count().show()
testData.groupBy("label").count().show()
