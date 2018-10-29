from pyspark import SparkConf, SparkContext
import collections
#spark=SparkSession.builder.appName('customers').getOrCreate()
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.regression import LinearRegression

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)
#reading the data file to rdd-line
lines = sc.textFile("Ecommerce_Customers.csv")
#applying map to rdd and spliting the line and taking field 2

#print("-----------------------",lines.take(5))

#features = ['Avg Session Length','Time on App','Time on Website','Length of Membership']
#label = 'Yearly Amount Spent'

data_features = lines.map(lambda x: [(k,x[k]) for k in x.keys()])

#schema = StructType([StructField("label",DoubleType(),True),StructField("features",VectorUDT(),True)])


#assembler = VectorAssembler(inputCols = ['Avg Session Length','Time on App','Time on Website','Length of Membership'],outputCol='features')

#output = assembler.transform(lines)

#train_data,test_data = data.randomSplit([0.7,0.3])
#lr = LinearRegression(labelCol ='Yearly Amount Spent' )
#lr_model = lr.fit(train_data)
print("-------------------------------------------------------------------------")
#test_results = lr_model.evaluate(test_data)
#print("************************************",test_results.rootMeanSquaredError)
print("------------------------",data.contains())

