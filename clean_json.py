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

datapath = 'dataset.json'

dataset = spark.read.format("json").option("inferSchema", "true").load(datapath)

#Creo una nuova colonna contenente la concatenazioni tra titolo della recensione e testo della recensione
dataset = dataset.withColumn("reviewTS",
  concat(dataset["summary"], lit(" "),dataset["reviewText"])).drop("helpful").drop("reviewerID").\
  drop("reviewerName").drop("reviewTime").drop("asin").drop("unixReviewTime")

print 'Dataset with 3 stars rank ' + str(dataset.count())

#Elimino le recensioni con 3 stelline
dataset = dataset.filter("overall !=3")

print 'Dataset without 3 stars rank ' + str(dataset.count())
#dataset.show(10)

#Creo un Bucketizer per la suddivisione delle classi, da -1.0 a 4.0 (recensioni con 1.0, 2.0)
# da 4.0 a 6.0 (recensioni con 4.0, 5.0)
splits = [-1.0, 4.0, 6.0]
bucketizer = Bucketizer(splits=splits, inputCol="overall", outputCol="label")
dataset = bucketizer.transform(dataset)

#Ripulisco il dataset droppando le colonne che non mi servono, tengo solo label e testo
dataset.groupBy("overall","label").count().show()
dataset = dataset.drop('overall').drop('reviewText').drop('summary')

#Calcolo rapporto tra classe positiva e negativa da utilizzare per l'undersampling
ratio = dataset.filter(dataset["label"] == 0.0).count() / dataset.filter(dataset["label"] == 1.0).count()

#Applico l'undersampler utilizzando il rapporto tra le classi calcolato in precedenza
dataset = dataset.sampleBy("label", fractions={1.0: ratio, 0.0: 1.0}, seed=42)
dataset.groupBy("label").count().show()

#Splitto il dataset in training_set e test_set tenendo 80% per training_set e 20% test_set
[training_data, test_data] = dataset.randomSplit([0.8, 0.2], 42)

print 'Dataset dimendion ' + str(training_data.count())
print 'Testset dimendion ' + str(test_data.count())
dataset.show(10)

#Salvo i nuovi json
training_data.write.mode('overwrite').json('training_set_BIG')
test_data.write.mode('overwrite').json('test_set_BIG')

spark.stop()
