package org.apache.spark.mllib.util

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.{LassoModel, LassoWithSGD, LabeledPoint}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.LabeledPoint

object PerformanceTest {

  private val numLocalIterations = 1

  class LogisticRegressionLocalUpdate(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
    extends LogisticRegressionWithSGD {
    override val optimizer = new GradientDescentWithLocalUpdate(new LogisticGradient(), new SimpleUpdater())
        .setStepSize(localstepSize)
        .setNumIterations(localnumIterations)
        .setNumLocalIterations(numLocalIterations)
        .setRegParam(localregParam)
        .setMiniBatchFraction(localminiBatchFraction)
  }

  object LogisticRegressionLocalUpdate {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : LogisticRegressionModel =
      new LogisticRegressionLocalUpdate(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  class LogisticRegressionLargeMemory(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
    extends LogisticRegressionWithSGD {
    override val optimizer = new GradientDescentWithLargeMemory(new LogisticGradient(), new SimpleUpdater())
        .setStepSize(localstepSize)
        .setNumIterations(localnumIterations)
        .setNumLocalIterations(numLocalIterations)
        .setRegParam(localregParam)
        .setMiniBatchFraction(localminiBatchFraction)
  }

  object LogisticRegressionLargeMemory {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : LogisticRegressionModel =
      new LogisticRegressionLargeMemory(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  class SVMLocalUpdate(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
    extends SVMWithSGD {
    override val optimizer = new GradientDescentWithLocalUpdate(new HingeGradient(), new SquaredL2Updater())
        .setStepSize(localstepSize)
        .setNumIterations(localnumIterations)
        .setNumLocalIterations(numLocalIterations)
        .setRegParam(localregParam)
        .setMiniBatchFraction(localminiBatchFraction)
  }

  object SVMLocalUpdate {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : SVMModel =
      new SVMLocalUpdate(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  class SVMLargeMemory(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
    extends SVMWithSGD {
    override val optimizer = new GradientDescentWithLargeMemory(new HingeGradient(), new SquaredL2Updater())
        .setStepSize(localstepSize)
        .setNumIterations(localnumIterations)
        .setNumLocalIterations(numLocalIterations)
        .setRegParam(localregParam)
        .setMiniBatchFraction(localminiBatchFraction)
  }

  object SVMLargeMemory {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : SVMModel =
      new SVMLargeMemory(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

   class LassoWithSGDAnother(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
  extends LassoWithSGD {
    def this() = this(1.0, 100, 1.0, 1.0)
    @transient override val optimizer = new GradientDescentAnother(new LeastSquaresGradient(), new L1Updater())
      .setStepSize(localstepSize)
      .setNumIterations(localnumIterations)
      .setRegParam(localregParam)
      .setMiniBatchFraction(localminiBatchFraction)
  }

  object LassoWithSGDAnother {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : LassoModel =
      new LassoWithSGDAnother(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  class LassoLocalUpdate(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
  extends LassoWithSGD {
    def this() = this(1.0, 100, 1.0, 1.0)
    @transient override val optimizer = new GradientDescentWithLocalUpdate(new LeastSquaresGradient(), new L1Updater())
      .setStepSize(localstepSize)
      .setNumIterations(localnumIterations)
      .setNumLocalIterations(numLocalIterations)
      .setRegParam(localregParam)
      .setMiniBatchFraction(localminiBatchFraction)
  }

  object LassoLocalUpdate {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : LassoModel =
      new LassoLocalUpdate(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  class LassoLargeMemory(
      var localstepSize: Double,
      var localnumIterations: Int,
      var localregParam: Double,
      var localminiBatchFraction: Double)
  extends LassoWithSGD {
    def this() = this(1.0, 100, 1.0, 1.0)
    @transient override val optimizer = new GradientDescentWithLargeMemory(new LeastSquaresGradient(), new L1Updater())
      .setStepSize(localstepSize)
      .setNumIterations(localnumIterations)
      .setNumLocalIterations(numLocalIterations)
      .setRegParam(localregParam)
      .setMiniBatchFraction(localminiBatchFraction)
  }

  object LassoLargeMemory {
    def train(input: RDD[LabeledPoint],
              numIterations: Int,
              stepSize: Double,
              miniBatchFraction: Double = 1.0)
    : LassoModel =
      new LassoLargeMemory(stepSize, numIterations, 0.0, miniBatchFraction).run(input)
  }

  def main(args: Array[String]) {
    if (args.length != 5) {
      println("Usage: PerformanceTest <master> <input_dir> <step_size> " +
        "<niters> <mini_splits>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "PerformanceTest")

    var begin = System.nanoTime
    val data = MLUtils.loadLibSVMData(sc, args(1))
    var end = System.nanoTime
    println(s"read data ${(end-begin)/10.0e9}")

    /*
    begin = System.nanoTime
    var model = LogisticRegressionWithSGD.train(data, args(3).toInt * 10, args(2).toDouble)
    end = System.nanoTime
    println(s"LR original total ${(end-begin)/10.0e9}")

    begin = System.nanoTime
    model = LogisticRegressionLocalUpdate.train(data, args(3).toInt, args(2).toDouble)
    end = System.nanoTime
    println(s"LR local update total ${(end-begin)/10.0e9}")

    begin = System.nanoTime
    model = LogisticRegressionLargeMemory.train(data, args(3).toInt, args(2).toDouble)
    end = System.nanoTime
    println(s"LR large memory total ${(end-begin)/10.0e9}")

    //------------------------------------------------------------------------------------------

    begin = System.nanoTime
    var model2 = SVMWithSGD.train(data, args(3).toInt * 10, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"SVM original total ${(end-begin)/10.0e9}")

    begin = System.nanoTime
    model2 = SVMLocalUpdate.train(data, args(3).toInt, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"SVM local update total ${(end-begin)/10.0e9}")

    begin = System.nanoTime
    model2 = SVMLargeMemory.train(data, args(3).toInt, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"SVM large memory total ${(end-begin)/10.0e9}")
    */

    //-------------------------------------------------------------------------------------------

    begin = System.nanoTime
    var model3 = LassoWithSGD.train(data, args(3).toInt * 10, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"Lasso original total ${(end-begin)/10.0e9}")

    /*
    begin = System.nanoTime
    model3 = LassoLocalUpdate.train(data, args(3).toInt, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"Lasso local update total ${(end-begin)/10.0e9}")

    begin = System.nanoTime
    model3 = LassoLargeMemory.train(data, args(3).toInt, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"Lasso large memory total ${(end-begin)/10.0e9}")
    */

    begin = System.nanoTime
    model3 = LassoWithSGDAnother.train(data, args(3).toInt * 10, args(2).toDouble, 1.0)
    end = System.nanoTime
    println(s"Lasso original total ${(end-begin)/10.0e9}")
    sc.stop()
  }
}
