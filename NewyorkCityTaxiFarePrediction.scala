package com.mapr.mlib;

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.log4j.{ Logger }
import org.apache.spark.sql.types.{ IntegerType, FloatType, DateType }
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.tuning.{ ParamGridBuilder, CrossValidator }
import org.apache.spark.ml.feature.{ Imputer, ImputerModel }
import org.apache.spark.sql.{ SQLContext, Row, DataFrame, Column }
import org.apache.spark.ml.feature.{ OneHotEncoder, StringIndexer, IndexToString }
import org.apache.spark.ml.feature.{ VectorAssembler, VectorIndexer }
import org.apache.spark.ml.feature.{ Bucketizer, Normalizer }
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import ml.dmlc.xgboost4j.scala.spark.{ XGBoostClassifier, XGBoostClassificationModel }
import scala.collection.mutable
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{ DoubleType, IntegerType, StringType, StructField, StructType }
import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.feature.VectorAssembler



//  val sc =trainingDF.drop("any");
//>>> taxiFence1 = taxi_df.filter(taxi_df.pickup_lat > 40.7).filter(taxi_df.pickup_lat < 40.86) # correct result
//taxiFence1.count()
//232496L
//taxiFence2 = taxi_df.filter("pickup_lat > 40.7 and pickup_lat < 40.86") # correct result
//taxiFence2.count()
//232496L
//taxiFence3 = taxi_df.filter(taxi_df.pickup_lat > 40.7 and taxi_df.pickup_lat < 40.86)  # wrong result
//taxiFence3.count()
//249896L
//taxiFence4 = taxi_df.filter(taxi_df.pickup_lat < 40.86) # same condition w/ the 2nd one above (and same result)
//taxiFence4.count()
//249896L
//df.na.drop(["onlyColumnInOneColumnDataFrame"]).
//df.na.drop(how = 'any')
//df.agg(min($"A"), min($"B")).first().toSeq.map{ case x: Int => x }.min
//val max_min = trainingDF.filter("pickup_latitude > "+lat_min+" and pickup_latitude < "+lat_max+"")
//                        .filter("dropoff_latitude > "+lat_min+"  and dropoff_latitude < "+lat_max+"")
//                        .filter("pickup_longitude > "+lon_min+" and pickup_longitude < "+lon_max+"")
//                        .filter("dropoff_longitude > "+lon_min+" and dropoff_longitude < "+lon_max+"")


object NewyorkCityTaxiFarePrediction extends App {

  @transient lazy val logger = Logger.getLogger(getClass.getName)

  val name = "City Taxi Fare Prediction"
  logger.info(s"Starting up $name")

  //    val conf = new SparkConf().setAppName(name).setMaster("local[*]").set("spark.cores.max", "2")
  //    val sc = new SparkContext(conf)
  val spark: SparkSession = SparkSession.builder().master("local[*]").appName(name).getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val schemaStruct = StructType(
    StructField("key", DateType) ::
      StructField("fare_amount", FloatType) ::
      StructField("pickup_datetime", DateType) ::
      StructField("pickup_longitude", FloatType) ::
      StructField("pickup_latitude", FloatType) ::
      StructField("dropoff_longitude", FloatType) ::
      StructField("dropoff_latitude", FloatType) ::
      StructField("passenger_count", IntegerType) :: Nil)
      
      val schemaTestStruct = StructType(
    StructField("key", DateType) ::      
      StructField("pickup_datetime", DateType) ::
      StructField("pickup_longitude", FloatType) ::
      StructField("pickup_latitude", FloatType) ::
      StructField("dropoff_longitude", FloatType) ::
      StructField("dropoff_latitude", FloatType) ::
      StructField("passenger_count", IntegerType) :: Nil)      
    
  import spark.implicits._

  val trainingDF = spark.read
    .option("header", "true")
    .schema(schemaStruct)
    .format("csv")
    .load("hdfs://localhost:8020/spark/mlib/xgboostclassifier/citytaxifare/train.csv").na.drop();

  trainingDF.show();

//  trainingDF.describe("fare_amount", "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count").show()
//
//  import spark.implicits._
//
 val dfTaxiSubset = trainingDF.filter("fare_amount > 0 and fare_amount <= 300")
////  |key|fare_amount|pickup_datetime|pickup_longitude|pickup_latitude|dropoff_longitude|dropoff_latitude|passenger_count|
////+---+-----------+---------------+----------------+---------------+-----------------+----------------+---------------+
////+---+-----------+---------------+----------------+---------------+-----------------+----------------+---------------+
  val test_data = spark.read
    .option("header", "true")
    .schema(schemaTestStruct)
    .format("csv")
    .load("hdfs://localhost:8020/spark/mlib/xgboostclassifier/citytaxifare/test.csv").na.drop();
//  
//  test_data.show();
//var lon_min = test_data.agg(min($"pickup_longitude"), min($"dropoff_longitude")).first().toSeq.map{ case x: Float => x }.min;
//var lon_max = test_data.agg(max($"pickup_longitude"), max($"dropoff_longitude")).first().toSeq.map{ case x: Float => x }.max;
//
//println(lon_min+"  lon_min");
//println(lon_max+"  lon_max");
//
//val lat_min=test_data.agg(min($"pickup_latitude"), min($"dropoff_latitude")).first().toSeq.map{ case x: Float => x }.min;
//val lat_max=test_data.agg(max($"pickup_latitude"), max($"dropoff_latitude")).first().toSeq.map{ case x: Float => x }.max;
//
//println(lat_min+"  lat_min");
//println(lat_max+"  lat_max");
//
//
 
   
  
var lon_min = dfTaxiSubset.agg(min($"pickup_longitude"), min($"dropoff_longitude")).first().toSeq.map{ case x: Float => x }.min;
var lon_max = dfTaxiSubset.agg(max($"pickup_longitude"), max($"dropoff_longitude")).first().toSeq.map{ case x: Float => x }.max;

println(lon_min+"  lon_min");
println(lon_max+"  lon_max");

val lat_min=dfTaxiSubset.agg(min($"pickup_latitude"), min($"dropoff_latitude")).first().toSeq.map{ case x: Float => x }.min;
val lat_max=dfTaxiSubset.agg(max($"pickup_latitude"), max($"dropoff_latitude")).first().toSeq.map{ case x: Float => x }.max;

println(lat_min+"  lat_min");
println(lat_max+"  lat_max");


val max_min = dfTaxiSubset.filter(s"pickup_latitude > $lat_min and pickup_latitude < $lat_max")
                        .filter(s"dropoff_latitude > $lat_min and dropoff_latitude < $lat_max")
                        .filter(s"pickup_longitude > $lon_min and pickup_longitude < $lon_max")
                        .filter(s"dropoff_longitude > $lon_min and dropoff_longitude < $lon_max")
    max_min.show(); 

 val removeOutliers = max_min.filter("passenger_count < 9").filter("passenger_count == 0");
removeOutliers.limit(100).show();
  // val passengerCount = removeOutliers.filter("passenger_count == 0")
  //println(passengerCount.count());
////val outliers = df.filter(s"value < $lowerRange or value > $upperRange")
//
//
//val someRows = max_min.limit(100);
//println(max_min.count());
//someRows.show();


 val dfWithDistance=removeOutliers
  .withColumn("a", pow(sin(toRadians($"dropoff_latitude" - $"pickup_latitude") / 2), 2) + cos(toRadians($"pickup_latitude")) * cos(toRadians($"dropoff_latitude")) * pow(sin(toRadians($"dropoff_longitude" - $"pickup_longitude") / 2), 2))
  .withColumn("distance", atan2(sqrt($"a"), sqrt(-$"a" + 1)) * 2 * 6371)
  
 
  
   
  val dataNonZeroDistance = dfWithDistance.filter("distance != 0");
  
  val dataFrameTransferred =dataNonZeroDistance.withColumn("pickupyear", year($"pickup_datetime")) 
                                       .withColumn("pickupmonth", month($"pickup_datetime"))
                                       .withColumn("pickupdate", dayofmonth($"pickup_datetime"))  
                                       .withColumn("pickupdayofweek", weekofyear($"pickup_datetime"));
  
  dataFrameTransferred.show();
  
  
dataFrameTransferred.createOrReplaceTempView("citytaxi")

    println("Check the pickup date and time affect the fare or not")
    spark.sql("select pickupdate, fare_amount from citytaxi  order by pickupdate").show

println("number of passengers vs fare");

spark.sql("select passenger_count, fare_amount from citytaxi  order by passenger_count").show

println("Does the day of the week affect the fare?");

spark.sql("select pickupdayofweek, fare_amount from citytaxi  order by fare_amount").show


val test_dataTransferred =test_data.withColumn("pickupyear", year($"pickup_datetime")) 
                                       .withColumn("pickupmonth", month($"pickup_datetime"))
                                       .withColumn("pickupdate", dayofmonth($"pickup_datetime"))  
                                       .withColumn("pickupdayofweek'", weekofyear($"pickup_datetime"));

val test_dataWithDistance=test_data
  .withColumn("a", pow(sin(toRadians($"dropoff_latitude" - $"pickup_latitude") / 2), 2) + cos(toRadians($"pickup_latitude")) * cos(toRadians($"dropoff_latitude")) * pow(sin(toRadians($"dropoff_longitude" - $"pickup_longitude") / 2), 2))
  .withColumn("distance", atan2(sqrt($"a"), sqrt(-$"a" + 1)) * 2 * 6371)
  
val newTrainDF = dataFrameTransferred.drop("pickup_datetime")
val newTest_data = test_dataWithDistance.drop("pickup_datetime").drop("key");
val testAssembler =  new VectorAssembler()
                  .setInputCols(Array("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count","distance"))
                  .setOutputCol("features")
                  
 val testXgbInput = testAssembler.transform(newTest_data);                 

//X=data.drop("fare_amount",axis=1)
//y=data.iloc[:,0].values


val assembler =  new VectorAssembler()
                  .setInputCols(Array("pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count","distance"))
                  .setOutputCol("features")

val xgbInput = assembler.transform(newTrainDF).withColumnRenamed("fare_amount", "label")

val xgbParam = Map("eta" -> 0.3,
      "max_depth" -> 15,
      "objective" -> "reg:linear",
      "num_round" -> 10,
      "num_workers" -> 2)
      
val xgbRegressor = new XGBoostRegressor(xgbParam).setFeaturesCol("features").setLabelCol("label")
val xgbRegressionModel = xgbRegressor.fit(xgbInput);
println(xgbRegressionModel.labelCol);
val pred = xgbRegressionModel.transform(testXgbInput);
pred.show();

}
