import sys 
from pyspark import SparkContext 
from pyspark.sql import SQLContext

sparkc = SparkContext("local", "task-1")
sqlc = SQLContext(sparkc)

