from pyspark import SparkContext 
from pyspark.sql.session import SparkSession 
from pyspark.streaming import StreamingContext 
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression as logreg
from pyspark.sql import Row 
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.ml.feature import StopWordsRemover, HashingTF, Tokenizer, IDF, CountVectorizer, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator as evalmet
from pyspark.ml.classification import RandomForestClassifier as rfclass
from pyspark.ml.classification import DecisionTreeClassifier as dtclass
from pyspark.ml.feature import MinMaxScaler
import pandas as pd
import pickle
from sklearn.linear_model import SGDClassifier as sgdc, PassiveAggressiveClassifier as pac
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score 
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.model_selection import train_test_split
from pyspark.mllib.linalg.distributed import RowMatrix
from sklearn.metrics.pairwise import pairwise_distances_argmin


schema = StructType([
	StructField("Sentiment",IntegerType(),False),
	StructField("Tweet",StringType(),False)
	])


def preproc(words):
		
	rdds=words.collect()
	values=list()
	for j in rdds:
		values=[i for i in list(json.loads(j).values())]
	
	if len(values)==0:
		return 
	df=spark.createDataFrame((Row(**d) for d in values), schema)
	
	df.na.drop()
	replace=['@\w+','#','RT',':']
	for r in replace:
		df=df.withColumn('Tweet',F.regexp_replace('Tweet',r,''))
	df=df.withColumn('Tweet',F.regexp_replace('Tweet',r'http\S+',''))
	#(train, test) = df.randomSplit([0.7,0.3])

	
	def minipreproc(df):
		tokenizer=Tokenizer(inputCol='Tweet', outputCol='Tweettoks')
		tokenizerdata=tokenizer.transform(df)
		
		swr=StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol='nostopwords')
		swrdata=swr.transform(tokenizerdata)
		
		hashtrans=HashingTF(numFeatures=2**9,inputCol=swr.getOutputCol(), outputCol='tweetfeat')
		hashdata=hashtrans.transform(swrdata)
		
		idf=IDF(inputCol=hashtrans.getOutputCol(), outputCol='tfidfcol')
		idfdata=idf.fit(hashdata)
		featuresdata=idfdata.transform(hashdata)
		_stringIdx = StringIndexer(inputCol = "Sentiment", outputCol = "label")
		stringdata=_stringIdx.fit(featuresdata).transform(featuresdata)

		return stringdata
	
	df=minipreproc(df)
	
	(train, test) = df.randomSplit([0.8,0.2], seed=3500)
	
	#features=["tfidfcol","Sentiment"]
	
	#finaldata=featuresdata[features]

	#x_train, x_test, y_train, y_test = train_test_split(x, y)'''
	
	feature=np.array(train.select("tfidfcol").collect()).reshape(-1,512)
	label=np.array(train.select("Sentiment").collect()).reshape(-1)
	testfeature=np.array(test.select('tfidfcol').collect()).reshape(-1,512)
	testlabel=np.array(test.select('Sentiment').collect()).reshape(-1)
	
	#LOGISTIC REGRESSION USING SGD CLASSIFIER
	#initialize model 
	logregsgd=sgdc(loss='log')
	#partial fit 
	logregsgd.partial_fit(feature,label,classes=[0,4])
	logregfile="logregmodel.pkl"
	#save model on train
	pickle.dump(logregsgd, open(logregfile,"wb"))
	#load model on test 
	logregloaded=pickle.load(open(logregfile,"rb"))
	yhatlr=logregloaded.predict(testfeature)
	scorelr=accuracy_score(yhatlr,testlabel)
	f1lr=f1_score(yhatlr, testlabel, average='weighted')
	plr=precision_score(yhatlr, testlabel, average='weighted')
	#print("SGDClassifier with Logistic Regression:",scorelr)

	
	#NAIVE BAYES MULTINOMIAL 
	multinb=MultinomialNB(alpha=0.4, fit_prior=True)
	multinb.partial_fit(feature, label, classes=[0,4])
	nb="naivebayes.pkl"
	pickle.dump(multinb, open(nb,"wb"))
	
	nbloaded=pickle.load(open(nb,"rb"))
	yhatnb=nbloaded.predict(testfeature)
	scorenb=accuracy_score(yhatnb,testlabel)
	f1nb=f1_score(yhatnb, testlabel, average='weighted')
	pnb=precision_score(yhatnb, testlabel, average='weighted')
	#print("Naive Bayes Multinomial NB:",scorenb)

	
	#PAC
	pacmodel=pac(C=0.7,random_state=100,shuffle=False)
	pacmodel.partial_fit(feature, label,classes=[0,4])
	pacfile="pac.pkl"
	pickle.dump(pacmodel, open(pacfile,"wb"))
	
	pacloaded=pickle.load(open(pacfile,"rb"))
	yhatpac=pacloaded.predict(testfeature)
	scorepac=accuracy_score(yhatpac, testlabel)
	f1pac=f1_score(yhatpac, testlabel, average='weighted')
	ppac=precision_score(yhatpac, testlabel, average='weighted')
	#print("PassiveAggressiveClassifier:", scorepac)

	#sgdcfile.write(f'{scorelr} {f1lr} {plr}\n')
	print(scorelr.round(3), scorenb.round(3), scorepac.round(3), f1lr.round(3), f1nb.round(3), f1pac.round(3), plr.round(3), pnb.round(3), ppac.round(3))
	
	#lrf1.write(f1lr)
	#lrpr.write(plr)
	
	#nbfile.write(f"{scorenb} {f1nb} {pnb}\n")
	#nbf1.write(f1nb)
	#nbpr.write(pnb)
	
	#pacfile.write(f"{scorepac} {f1pac} {ppac}\n")
	#pacf1.write(f1pac)`
	#pacpr.write(ppac)
	
	
	#print(scorelr, scorenb, scorepac)
	#print(f1lr, f1nb, f1pac)
	#print(plr, pnb, ppac)
	#print('\n')
	
	#MINIBATCH KMEANS
	kmeans=MiniBatchKMeans(n_clusters=2, init='k-means++',random_state=100)
	kmeans=kmeans.partial_fit(feature)
	preds=kmeans.predict(testfeature)
	centroids=kmeans.cluster_centers_
	
	'''kmeanslabels=pairwise_distances_argmin(feature,centroids)
	colors=["#4EACC5","#ff9c34"]
	for k,col in zip(range(2),colors):
		members=kmeanslabels==k
		cluster_center=centroids[k]
		plt.plot(feature[members,0],feature[members,1],"w",markerfacecolor=col)
		plt.show()
	
	#plt.scatter(centroids[:,0],centroids[:,1],s=80,color='black')
	
	#plt.show()
	
	print(preds)'''
	
	#feat_minmax=MinMaxScaler(inputCol=hashtrans.getOutputCol(), outputCol='scaledfeat')
	#preprocessed=hashdata.select('tweetfeat','tweet','sentiment')
	#preprocessed.show()
	#print(preprocessed)
	#df.show()'''
	
	'''model=logreg(featuresCol='tweetfeat',labelCol='Sentiment')
	#rf=rfclass(labelCol='Sentiment',featuresCol='tweetfeat'mapdepth=6)
	pipeline1=Pipeline(stages=[tokenizer,swr,hashtrans,model])
	
	pipefit1=pipeline1.fit(train)
	pred1=pipefit1.transform(test)
	final=pred1.select('Tweet','Sentiment','prediction')
	final.show()
	evaluator1=evalmet(labelCol='Sentiment', predictionCol='prediction', metricName='accuracy')
	accuracy1=evaluator1.evaluate(pred1)
	print("logreg", accuracy1)
	
	rfmodel=rfclass(labelCol='Sentiment', featuresCol='tweetfeat')
	pipeline2=Pipeline(stages=[tokenizer,swr,hashtrans,rfmodel])
	pipefit2=pipeline2.fit(train)
	pred2=pipefit2.transform(test)
	evaluator2=evalmet(labelCol='Sentiment', predictionCol='prediction', metricName='accuracy')
	accuracy2=evaluator2.evaluate(pred2)
	print("randomforest", accuracy2)
	
	dtmodel=dtclass(labelCol='Sentiment', featuresCol='tweetfeat')
	pipeline3=Pipeline(stages=[tokenizer,swr,hashtrans,dtmodel])
	pipefit3=pipeline3.fit(train)
	pred3=pipefit3.transform(test)
	evaluator3=evalmet(labelCol='Sentiment', predictionCol='prediction', metricName='accuracy')
	accuracy3=evaluator3.evaluate(pred3)
	print("decisiontree", accuracy3)'''
	
	
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


