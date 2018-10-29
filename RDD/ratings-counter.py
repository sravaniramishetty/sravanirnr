from pyspark import SparkConf, SparkContext
import collections

conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf = conf)
#reading the data file to rdd-line
lines = sc.textFile("u.data")
#applying map to rdd and spliting the line and taking field 2


ratings = lines.map(lambda x: x.split()[2])
#counting the number of values to the keys
result = ratings.countByValue()
#collectins to sort the key,vakue pairs
sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
	print("%s %i" % (key, value))
