from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F

def preproc(tweets):
	words=tweets.select(explode(split(tweets.value,"t_end")).alias("word"))
	words=words.na.replace('',None)
	words=words.na.drop()
	replace=['@\w+','#','RT',':','http\S+']
	for r in replace:
		words=words.withColumn('word',F.regexp_replace('word',r,''))
		
	return words
