from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import StopWordsRemover, HashingTF, RegexTokenizer 

def preproc(tweets):
	words=tweets.select(explode(split(tweets.value,"t_end")).alias("Tweet"))
	words=words.na.replace('',None)
	words=words.na.drop()
	replace=['@\w+','#','RT',':']
	for r in replace:
		words=words.withColumn('Tweet',F.regexp_replace('Tweet',r,''))
	words=words.withColumn('Tweet',F.regexp_replace('Tweet',r'http\S+',''))
	
	tokenize=Tokenizer(inputCol='Tweet',outputCol='Tweettoks')
	tokendata=tokenize.transform(words)
	
	swr=StopWordsRemover(inputCol=tokenize.getOutputCol(),outputCol='nostopw')
	swrdata=swr.transform(tokendata)
	
	hashtrans=HashingTF(inputCol=swr.getOutputCol(),outputCol='tweetfeat')
	hashdata=hashtrans.transform(swrdata)
	
	preprocdata = hashdata.select('tweetfeat','Tweet')
	
	return preprocdata
