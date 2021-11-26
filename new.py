from pyspark import SparkContext 
from pyspark.sql.session import SparkSession 
from pyspark.streaming import StreamingContext 
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as logreg
from pyspark.sql import Row 
import json
import sys
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import StopWordsRemover, HashingTF, Tokenizer 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as evalmet
from pyspark.ml.classification import RandomForestClassifier as rfclass


schema = StructType([
	StructField("Sentiment",IntegerType(),False),
	StructField("Tweet",StringType(),False)
	])
	
def preproc(words):
	rdds=words.collect()
	values=[i for j in rdds for i in list(json.loads(j).values())]
	
	if len(values)==0:
		return 
	df=spark.createDataFrame((Row(**d) for d in values), schema)
	
	df.na.drop()
	replace=['@\w+','#','RT',':']
	for r in replace:
		df=df.withColumn('Tweet',F.regexp_replace('Tweet',r,''))
	df=df.withColumn('Tweet',F.regexp_replace('Tweet',r'http\S+',''))
	(train, test) = df.randomSplit([0.7,0.3])
	
	
	tokenizer=Tokenizer(inputCol='Tweet', outputCol='Tweettoks')
	#tokenizerdata=tokenizer.transform(df)
	
	swr=StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='nostopwords')
	#swrdata=swr.transform(tokenizerdata)
	
	hashtrans=HashingTF(inputCol=swr.getOutputCol(), outputCol='tweetfeat')
	#hashdata=hashtrans.transform(swrdata)
	
	#preprocessed=hashdata.select('tweetfeat','tweet','sentiment')
	#preprocessed.show()
	#print(preprocessed)
	#df.show()
	
	model=logreg(featuresCol='tweetfeat',labelCol='Sentiment')
	#rf=rfclass(labelCol='Sentiment',featuresCol='tweetfeat'mapdepth=6)
	pipeline=Pipeline(stages=[tokenizer,swr,hashtrans,model])
	
	pipefit=pipeline.fit(train)
	pred=pipefit.transform(test)
	pred.show()
	evaluator=evalmet(labelCol='Sentiment', predictionCol='prediction', metricName='accuracy')
	
	accuracy=evaluator.evaluate(pred)
	
	print(accuracy)
	#pred=pipefit.transform(df).select('sentiment','prediction').show()
	"""try:
		text = words.filter(lambda x: len(x) > 0)
		rowrdd=text.map(lambda w: Row(tweet=w))
		print(rowrdd)
		wordsdf=spark.createDataFrame(text, schema)
		wordsdf.show()
	except:
		print('None')"""

	
	
sc = SparkContext()
spark=SparkSession(sc)

ssc=StreamingContext(sc,1)

lines=ssc.socketTextStream("localhost",6100)

words=lines.flatMap(lambda line: line.split("\n"))
words.foreachRDD(preproc)

ssc.start()
ssc.awaitTermination()


