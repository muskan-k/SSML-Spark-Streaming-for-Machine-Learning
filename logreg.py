from pyspark import SparkContext 
from pyspark.sql.session import SparkSession 
from pyspark.streaming import StreamingContext 
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, StopWordsRemover, Word2Vec, RegexTokenizer 
from pyspark.ml.classification import LogisticRegression as logreg
from pyspark.sql import Row 

sc=SparkContext(appname="project")
spark=SparkSession(sc)

data=spark.read.csv('train.csv',inferSchema=True,header=True)

stage_1= RegexTokenizer(inputCol='Tweet', outputCol='tokens', pattern='\\W')
stage_2=StopWordsRemover(inputCol='tokens', outputCol='filtered')
stage_3=Word2Vec(inputCol='filtered', outputCol='vector', vectorSize=100)
model=logreg(featuresCol='vector',labelCol='Sentiment')

pipeline=Pipeline(stages=[stage_1,stage_2,stage_3,model])

pipefit=pipeline.fit(data)

#TO DO: streaming into model
