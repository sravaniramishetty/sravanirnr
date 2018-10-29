////////////////////////////////////////////
//// LINEAR REGRESSION EXERCISE ///////////
/// Complete the commented tasks below ///
/////////////////////////////////////////

// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Use Spark to read in the Ecommerce Customers csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Ecommerce_Customers.csv")

// Print the Schema of the DataFrame
data.printSchema

// Print out an example Row
// Various ways to do this, just
// choose whichever way you prefer


////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// A few things we need to do before Spark can accept the data!
// It needs to be in the form of two columns
// ("label","features")

// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Rename the Yearly Amount Spent Column as "label"
// Also grab only the numerical columns from the data
// Set all of this as a new dataframe called df
val df = data.select(data("Yearly Amount Spent").as("label"),$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")

val assembler = (new VectorAssembler().setInputCols(Array("Avg Session Length", 
                 "Time on App", "Time on Website", "Length of Membership")).setOutputCol("features"))

val output = assembler.transform(df).select($"label",$"features")

val lr = new LinearRegression()

val lrModel = lr.fit(output)

println(s"Coeff:${lrModel.coefficients},intercept:${lrModel.intercept}")

val trainingSummary = lrModel.summary
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"R2: ${trainingSummary.r2}")


// An assembler converts the input values to a vector
// A vector is what the ML algorithm reads to train a model

// Use VectorAssembler to convert the input columns of df
// to a single output column of an array called "features"
// Set the input columns from which we are supposed to read the values.
// Call this new object assembler

// Use the assembler to transform our DataFrame to the two columns: label and features


// Create a Linear Regression Model object


// Fit the model to the data and call this model lrModel


// Print the coefficients and intercept for linear regression

// Summarize the model over the training set and print out some metrics!
// Use the .summary method off your model to create an object
// called trainingSummary

// Show the residuals, the RMSE, the MSE, and the R^2 Values.


// Great Job!
