import time
start = time.time()

from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('customers').getOrCreate()
from pyspark.ml.regression import LinearRegression

data=spark.read.csv('Ecommerce_Customers.csv',inferSchema = True,header=True)

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols = ['Avg Session Length','Time on App','Time on Website','Length of Membership'],outputCol='features')

output = assembler.transform(data)

final_data = output.select('features','Yearly Amount Spent')

train_data,test_data = final_data.randomSplit([0.7,0.3])

lr = LinearRegression(labelCol ='Yearly Amount Spent' )

lr_model = lr.fit(train_data)
print("-------------------------------------------------------------------------")
test_results = lr_model.evaluate(test_data)

print("************************************",test_results.rootMeanSquaredError)

print(test_results.meanSquaredError)

print(test_results.r2)

end = time.time()
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ",end-start)




