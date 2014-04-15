/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
class GradientDescentWithLocalUpdate(var gradient: Gradient, var updater: Updater) extends GradientDescent(gradient, updater) with Logging {
  private var numLocalIterations: Int = 1
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  override def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  override def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  override def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  override def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  override def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  override def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Set the number of local iterations. Default 1.
   */
  def setNumLocalIterations(numLocalIter: Int): this.type = {
    this.numLocalIterations = numLocalIter
    this
  }

  override def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {

    val (weights, _) = GradientDescentWithLocalUpdate.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      numLocalIterations,
      regParam,
      miniBatchFraction,
      20,
      initialWeights)
    weights
  }

}

// Top-level method to run gradient descent.
object GradientDescentWithLocalUpdate extends Logging {
   /**
   * Run gradient descent with local update in parallel using mini batches. Unlike the
   * [[GradientDescent]], here gradient descent takes place not only among jobs, but also inner
   * jobs, i.e. on an executor.
   *
   * @param data - Input data for SGD. RDD of form (label, [feature values]).
   * @param gradient - Gradient object that will be used to compute the gradient.
   * @param updater - Updater object that will be used to update the model.
   * @param stepSize - stepSize to be used during update.
   * @param numIterations - number of outer iterations that SGD should be run.
   * @param numLocalIterations - number of inner iterations that SGD should be run.
   * @param regParam - regularization parameter
   * @param miniBatchFraction - fraction of the input data set that should be used for
   *                            one iteration of SGD. Default value 1.0.
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the stochastic
   *         loss computed for every iteration.
   */
  def runMiniBatchSGD(
      data: RDD[(Double, Vector)],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      numLocalIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      tinyBatchSize: Int = 20,
      initialWeights: Vector): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)

     /**
     * For the first iteration, the regVal will be initialized as sum of sqrt of
     * weights if it's L2 update; for L1 update; the same logic is followed.
     */
    val regVal = updater.compute(
      weights, Vectors.dense(new Array[Double](weights.size)), 0, 1, regParam)._2

    for (i <- 1 to numIterations) {
      val weightsAndLosses = data.mapPartitions { iter =>
        val rand = new Random(42 + i * numIterations)
        val sampled = iter.filter(x => rand.nextDouble() <= miniBatchFraction)

        val ((weightsAvg, lossAvg, regValAvg), _) = sampled.grouped(tinyBatchSize).map {
          case tinyDataSets =>

            Iterator.iterate((weights, 0.0, regVal)) {
              case (lastWeights, lastLoss, lastRegVal) =>

                val (localGrad, localLoss) = tinyDataSets.map { case (y, featuresCol) =>
                  gradient.compute(featuresCol, y, lastWeights)
                }.reduce((a, b) => (Vectors.fromBreeze(a._1.toBreeze += b._1.toBreeze), a._2 + b._2))

                val grad = Vectors.fromBreeze(localGrad.toBreeze :/ tinyDataSets.size.toDouble)
                val loss = localLoss / tinyDataSets.size

                val (newWeights, newRegVal) = updater.compute(
                  lastWeights, grad, stepSize, i * numIterations, regParam)

                (newWeights, loss + lastLoss, newRegVal + lastRegVal)

            }.drop(numLocalIterations).next()

        }.foldLeft(((BDV.zeros[Double](initialWeights.size), 0.0, 0.0), 0.0)) {
          case (((lMatrix, lLoss, lRegVal), count), (rMatrix, rLoss, rRegVal)) =>
            var i = 0
            while (i < lMatrix.length) {
              lMatrix(i) *= count
              lMatrix(i) += rMatrix(i)
              lMatrix(i) /= (count + 1)
              i += 1
            }
            ((lMatrix,
              (lLoss * count + rLoss) / (count + 1),
              (lRegVal * count + rRegVal) / (count + 1)
              ),
              count + 1)
        }

        val localLossHistory = (lossAvg + regValAvg) / numLocalIterations

        List((weightsAvg, localLossHistory)).iterator
      }

      val c = weightsAndLosses.collect()
      val (ws, ls) = c.unzip

      stochasticLossHistory.append(ls.reduce(_ + _) / ls.size)

      weights = Vectors.fromBreeze(ws.reduce(_ += _) :/ c.size.toDouble)
    }

    logInfo("GradientDescentWithLocalUpdate finished. Last a few stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)
  }
}
