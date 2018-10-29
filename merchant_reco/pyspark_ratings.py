from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
import pyspark
from pyspark.sql import Row
import csv  

sql_c = SQLContext(sc)
spark = SparkSession.builder.appName("Basics").getOrCreate()

df = sql_c.read.csv('ratings.csv')
df.show()
