import java.io.File
import collection.JavaConverters._
import com.sigopt.Sigopt
import com.sigopt.model.{Assignments, Bounds, Experiment, Parameter, Suggestion, Observation}
import org.apache.spark.rdd._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object MovieLensExperiment {
  def createExperiment(name: String, parameters: Seq[(String, Double, Double, String)]) : Experiment = {
    val e: Experiment = Experiment.create()
      .data(new Experiment.Builder()
        .name(name)
        .parameters(parameters.map({ case (name, min, max, typ) => {
          new Parameter.Builder()
            .name(name)
            .`type`(typ)
            .bounds(new Bounds.Builder()
              .min(min)
              .max(max)
              .build())
            .build()}}).asJava)
        .build())
      .call()
    return e
  }

  def createObservation(experimentId: String, s: Suggestion, metric: Double): Observation = {
    new Experiment(experimentId).observations().create()
      .data(new Observation.Builder()
        .suggestion(s.getId())
        .value(metric)
        .build())
      .call()
  }

  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }

  def main(args: Array[String]): Unit ={
    // Find your SigOpt client token here : https://sigopt.com/user/profile
    Sigopt.clientToken = "<YOUR_CLIENT_TOKEN>"
    val conf = new SparkConf().setAppName("SigOpt Spark Test")
    val sc = new SparkContext(conf)

    // process ratings csv into RDD
    val movieLensHomeDir = "data/ml-latest"
    val rawdata = sc.textFile(new File(movieLensHomeDir, "ratings.csv").toString)
    val header  = rawdata.first()
    val tbldata = rawdata.filter(_(0) != header(0))
    val ratings = tbldata.map { line =>
      val fields = line.split(",")
      // format: (timestamp % 10, Rating(userId, movieId, rating))
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
    }

    // randomly split into train, valid, test sets
    val Array(training, valid, test) = ratings.randomSplit(Array(0.75, 0.1, 0.15), seed=42)
    val numValid = valid.count()
    val numTest = test.count()

    // create SigOpt experiment
    val experiment = createExperiment("SigOpt Search : ALS Recommender", Seq(
      ("rank", 2, 40, "int"),
      ("numIter", 3, 30, "int"),
      ("log_lambda", Math.log(0.0001), Math.log(100.0), "double"))
    )

    var bestAssignment = null.asInstanceOf[com.sigopt.model.Assignments]
    var bestValue = -1e12

    for (i <- 1 to 30) {
      val suggestion = new Experiment(experiment.getId).suggestions().create().call()
      val assgnmt = suggestion.getAssignments
      val rank = assgnmt.getInteger("rank")
      val numIter = assgnmt.getInteger("numIter")
      val lambda = Math.exp(assgnmt.getDouble("log_lambda"))
      val model = ALS.train(training, rank, numIter, lambda)

      // get prediction score
      val numTest = test.count()
      val validRmse = -1.0*computeRmse(model, valid, numValid)
      createObservation(experiment.getId, suggestion, validRmse)
      if (validRmse > bestValue) {
        bestValue = validRmse
        bestAssignment = assgnmt
      }
    }

    // fit on entire dataset ( train and valid ), using best parameters
    val model = ALS.train(sc.union(training,valid),
                          bestAssignment.getInteger("rank"), bestAssignment.getInteger("numIter"),
                          Math.exp(bestAssignment.getDouble("log_lambda")))

    // find rmse on test dataset
    val testRmse = computeRmse(model, test, numTest)

    // print results
    println("Best configuration RMSE : "+testRmse)

    sc.stop()
  }
}
